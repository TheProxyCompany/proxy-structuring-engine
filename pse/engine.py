from __future__ import annotations

import logging
from typing import Any

from lexpy import DAWG, Trie
from pse_core.walker import Walker
from pydantic import BaseModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.generation.logits_process import LogitsProcessor

from pse.acceptors.collections.encapsulated_acceptor import EncapsulatedAcceptor
from pse.state_machine import HierarchicalStateMachine
from pse.util.get_logit_bias import get_logit_bias
from pse.util.get_top_logits import get_top_logits
from pse.util.state_machine.get_acceptor import get_acceptor

logger = logging.getLogger(__name__)


class StructuringEngine(LogitsProcessor):
    """
    Drives a StateMachineAcceptor to manage and validate structured outputs based on a given schema.

    This driver utilizes various acceptors to ensure that the output adheres to the specified JSON schema
    or other supported schema types. It manages the state of token acceptance and provides mechanisms
    to advance tokens and characters while validating the structured output.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        vocabulary: dict[str, int] | None = None,
    ) -> None:
        """
        Initializes the StructuredOutputDriver with the provided tokenizer.

        Args:
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The tokenizer used to convert tokens to strqings.
            vocabulary (dict[str, int] | None): A dictionary mapping tokens to their IDs. Defaults to tokenizer's vocabulary.
        """
        StructuringEngine.build_vocabulary(tokenizer, vocabulary)
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self.state_machine: HierarchicalStateMachine | None = None
        self.walkers: list[Walker] = []
        self.within_json_value: bool = False

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """
        scores are logits
        """
        # Generate rows for each token
        rows = []
        top_logits = get_top_logits(scores, 20)
        valid_tokens, reversed_valid_tokens = self.get_valid_tokens()
        if not valid_tokens:
            return scores

        for token_id, score in top_logits.items():
            # Get token from token_id using reverse vocabulary map
            if not (token := self.reverse_vocabulary.get(token_id)):
                logger.warning(f"Unknown token ID: {token_id}")
                continue

            if token in valid_tokens:
                rows.append(f"{token_id:<8} | 🟢 {score:>9.4f} | {repr(token)[1:-1]}")
                continue

            if token not in valid_tokens:
                scores[token_id] = float("-inf")
                fixed_tokens = [
                    t[::-1]
                    for t in reversed_valid_tokens.search_with_prefix(token[::-1])
                    if isinstance(t, str)
                ]

                for fixed_token in fixed_tokens:
                    fixed_token_id = self.vocabulary[fixed_token]
                    scores[fixed_token_id] = score
                    rows.append(
                        f"{fixed_token_id:<8} | 🟢 {score:>9.4f} | {repr(fixed_token)[1:-1]}"
                    )

        header = f"{'Token ID':<8} | {'Score':>10} | Token"
        separator = "-" * 9 + "+" + "-" * 12 + "+" + "-" * 20
        chart = "\n".join([header, separator] + rows[:10])
        if rows:
            logger.info(f"🔵 Top logits:\n{chart}")
        else:
            logger.info("🔴 No valid tokens found")

        valid_token_ids = set(self.vocabulary[t] for t in valid_tokens)
        return scores + get_logit_bias(scores, valid_token_ids)

    def get_valid_tokens(self) -> tuple[set[str], Trie]:
        """
        Returns a list of valid token IDs based on the current state of the acceptor.

        Args:
            logits: The logits tensor to mask. Just used for dimensionality.

        Returns:
            The bias, of the same type as `logits`.
        """
        all_valid_prefixes = set()
        trie = Trie()
        for walker in self.walkers:
            if walker.accepts_any_token():
                return set(), trie

            # valid_prefixes = walker.find_valid_prefixes(self.dawg)
            all_valid_prefixes.update([])

        trie.add_all({s[::-1] for s in all_valid_prefixes})
        return all_valid_prefixes, trie

    def set_schema(
        self,
        schema: type[BaseModel]
        | list[type[BaseModel]]
        | dict[str, Any]
        | list[dict[str, Any]],
        use_delimiters: bool,
        delimiters: tuple[str, str] | None = None,
    ) -> None:
        """
        Adds a schema with delimiters to the engine.
        """

        def get_schema(schema: Any | None) -> dict[str, Any]:
            if schema is None:
                return {}

            if isinstance(schema, list):
                if schema and all(
                    isinstance(s, type) and issubclass(s, BaseModel) for s in schema
                ):
                    return {
                        "oneOf": [
                            s.model_json_schema()
                            for s in schema
                            if isinstance(s, type) and issubclass(s, BaseModel)
                        ]
                    }
                return {"oneOf": schema}
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_json_schema()
            if isinstance(schema, dict):
                if "schema" in schema:
                    logger.warning(
                        "Schema should not be provided as an object with 'schema' key."
                    )
                    return schema["schema"]
                return schema

            return {}

        acceptor = get_acceptor(
            schema=get_schema(schema),
            start_hook=self._start_hook,
            end_hook=self._end_hook,
        )

        if use_delimiters:
            open_delimiter, close_delimiter = delimiters or ("```json\n", "\n```")
            self.state_machine = EncapsulatedAcceptor(
                acceptor, open_delimiter, close_delimiter
            )
        else:
            self.state_machine = acceptor

        self.walkers = (
            list(self.state_machine.get_walkers()) if self.state_machine else []
        )

    def advance_token(self, token_id: int) -> int | None:
        """
        Advances the acceptor's state using the provided token ID.
        """
        if not (token := self.reverse_vocabulary.get(token_id)):
            logger.warning(f"Unknown token ID: {token_id}")
            return None

        seen: dict[str, set[Walker]] = {}
        longest_partial: tuple[str, int] = ("", -1)  # (partial_token, token_id)
        for valid_token, walker in HierarchicalStateMachine.advance_all(
            self.walkers, token, self.dawg
        ):
            seen.setdefault(valid_token, set()).add(walker)

            if valid_token != token:
                # Track longest partial (avoid sort operation later)
                if len(valid_token) > len(longest_partial[0]):
                    if valid_id := self.vocabulary.get(valid_token):
                        longest_partial = (valid_token, valid_id)

        if walkers := seen.get(token):
            self.walkers = list(walkers)
            return token_id

        if longest_partial[1] != -1:
            self.walkers = list(seen[longest_partial[0]])
            return longest_partial[1]

    def consume_raw_input(self, raw_input: str) -> None:
        """Advances the acceptor using the provided raw input.

        Breaks input into tokens and advances all walkers for each token.
        Only exact token matches are supported (no partial matches).

        Args:
            raw_input: The input string to advance the acceptor with.
        """
        # Process each token of the raw string input
        for token_id in self.tokenizer.encode(raw_input, add_special_tokens=False):
            token = self.tokenizer.decode([token_id])
            if not token:
                continue

            # Get walkers that accept this exact token
            new_walkers = [
                walker
                for valid_token, walker in HierarchicalStateMachine.advance_all(
                    self.walkers, token, self.dawg
                )
                if valid_token == token
            ]

            # Update walkers if we found valid transitions
            if new_walkers:
                self.walkers = new_walkers

    @property
    def in_structured_state(self) -> bool:
        """
        Checks whether the driver is in a structured state.

        If the acceptor is encapsulated, then the driver is not structured until the opening delimiter is triggered.
        If the acceptor is not encapsulated, then the driver is structured as soon as the first token is accepted.

        When processing input, if the driver is within a JSON value (i.e., has consumed the `"` character when processing
        a JSON string), then the driver is not structured until the closing delimiter is triggered. This allows us to
        enable/disable creativity sampling when inside JSON values within the JSON output, without having the creativity
        sampling affect the structured output generation.

        Returns:
            bool: True if in a structured state, False otherwise.
        """

        return not self._waiting_for_trigger() and not self.within_json_value

    @property
    def has_reached_accept_state(self) -> bool:
        """
        Checks whether the acceptor has reached a valid final state.

        Returns:
            bool: True if in an accepted state, False otherwise.
        """
        if not self.state_machine:
            return False

        return any(walker.has_reached_accept_state() for walker in self.walkers)

    @classmethod
    def build_vocabulary(
        cls,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        vocabulary: dict[str, int] | None = None,
    ) -> None:
        """
        Builds a vocabulary mapping for the tokenizer.

        Args:
            tokenizer: The tokenizer to build vocabulary from
            vocabulary: Optional custom vocabulary mapping. If not provided,
                       uses tokenizer's vocabulary.
        """
        cls.dawg = DAWG()
        cls.vocabulary: dict[str, int] = {}
        cls.reverse_vocabulary: dict[int, str] = {}

        # Get token IDs and decoded tokens
        vocab = vocabulary if vocabulary else tokenizer.get_vocab()
        token_ids = list(vocab.values())
        decoded_tokens = (
            list(vocab.keys()) if vocabulary else tokenizer.batch_decode(token_ids)
        )

        # Build DAWG from sorted tokens
        cls.dawg.add_all(sorted(decoded_tokens))
        cls.dawg.reduce()

        # Create token to ID mapping
        for token, id in zip(decoded_tokens, token_ids, strict=True):
            cls.vocabulary[token] = id
            cls.reverse_vocabulary[id] = token

    # -------- Private Methods --------

    def _waiting_for_trigger(self) -> bool:
        """
        Determines if the acceptor is waiting for the opening delimiter.

        Returns:
            bool: True if waiting for trigger, False otherwise.
        """
        if not self.state_machine or not isinstance(
            self.state_machine, EncapsulatedAcceptor
        ):
            return False

        return not any(walker.is_within_value() for walker in self.walkers)

    def _start_hook(self) -> None:
        """
        Called when the acceptor starts processing a new structured output.
        """
        self.within_json_value = True

    def _end_hook(self) -> None:
        """
        Called when the acceptor ends processing a structured output.
        """
        self.within_json_value = False
