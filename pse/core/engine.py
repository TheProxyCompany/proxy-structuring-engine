from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from lexpy import DAWG
from pydantic import BaseModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.generation.logits_process import LogitsProcessor

from pse.core.walker import Walker
from pse.core.state_machine import StateMachine
from pse.acceptors.collections.encapsulated_acceptor import EncapsulatedAcceptor
from pse.acceptors.basic.acceptor import Acceptor
from pse.util.errors import TokenRejected
from pse.util.state_machine.get_acceptor import get_acceptor
from pse.util.get_logit_bias import get_logit_bias
from pse.util.get_top_logits import get_top_logits
from pse.util.pydantic_to_json import pydantic_to_json

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
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        vocabulary: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initializes the StructuredOutputDriver with the provided tokenizer.

        Args:
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer used to convert tokens to strings.
            vocabulary (Optional[Dict[str, int]]): A dictionary mapping tokens to their IDs. Defaults to tokenizer's vocabulary.
        """
        StructuringEngine.build_vocabulary(tokenizer, vocabulary)
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = tokenizer
        self.eos_id: int = tokenizer.eos_token_id or 0
        self.acceptor: Optional[Acceptor] = None
        self.walkers: List[Walker] = []
        self.within_json_value: bool = False

    @classmethod
    def build_vocabulary(
        cls,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        vocabulary: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Builds a vocabulary mapping for the tokenizer.

        Args:
            tokenizer: The tokenizer to build vocabulary from
            vocabulary: Optional custom vocabulary mapping. If not provided,
                       uses tokenizer's vocabulary.
        """
        cls.dawg = DAWG()
        cls.vocabulary: Dict[str, int] = {}
        cls.reverse_vocabulary: Dict[int, str] = {}

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
        for token, id in zip(decoded_tokens, token_ids):
            cls.vocabulary[token] = id
            cls.reverse_vocabulary[id] = token

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

    def has_reached_accept_state(self) -> bool:
        """
        Checks whether the acceptor has reached a valid final state.

        Returns:
            bool: True if in an accepted state, False otherwise.
        """
        if not self.acceptor:
            return False

        return any(walker.has_reached_accept_state() for walker in self.walkers)

    def set_schema(
        self,
        schema: BaseModel | List[BaseModel] | Dict[str, Any] | List[Dict[str, Any]],
        use_delimiters: bool = True,
        delimiters: Optional[Tuple[str, str]] = None,
    ) -> None:
        """
        Adds a schema with delimiters to the engine.
        """

        def get_schema(schema: Any) -> Dict[str, Any]:
            if isinstance(schema, list):
                if schema and isinstance(schema[0], BaseModel):
                    return {"anyOf": [pydantic_to_json(type(s)) for s in schema]}
                return {"anyOf": schema}
            if isinstance(schema, BaseModel):
                return pydantic_to_json(type(schema))
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
            self.acceptor = EncapsulatedAcceptor(
                acceptor, open_delimiter, close_delimiter
            )
        else:
            self.acceptor = acceptor

        self.walkers = list(self.acceptor.get_walkers())

    def get_next_token(self, logprobs, top_k: int = 64) -> int:
        """
        Advances the acceptor's state using the provided logits.

        Args:
            logprobs: The log probabilities from the language model.

        Returns:
            int: The next token ID to generate.
        """
        # Single dict for both full and partial matches
        seen: Dict[str, Set[Walker]] = {}
        longest_partial: Tuple[str, int] = ("", -1)  # (partial_token, token_id)

        top_logits = get_top_logits(logprobs, top_k)
        for token_id, score in sorted(top_logits, key=lambda x: x[1], reverse=True):
            # Get token from token_id using reverse vocabulary map
            if not (token := self.reverse_vocabulary.get(token_id)):
                logger.warning(f"Unknown token ID: {token_id}")
                continue

            logger.debug(f"⚪️ LLM predicted token: {repr(token)}, Score: {score}")
            # Check if we've already seen this token (full match or partial match)
            if walkers := seen.get(token):
                self.walkers = list(walkers)
                return token_id

            # Advance state machine for this token
            for valid_token, walker in StateMachine.advance_all(
                self.walkers, token, self.dawg
            ):
                seen.setdefault(valid_token, set()).add(walker)

                if valid_token != token:
                    # Track longest partial (avoid sort operation later)
                    if len(valid_token) > len(longest_partial[0]):
                        if valid_id := self.vocabulary.get(valid_token):
                            longest_partial = (valid_token, valid_id)

            # If we advanced walkers for this token, return the token id
            if walkers := seen.get(token):
                self.walkers = list(walkers)
                return token_id

        # Fallback to the longest partial match
        if longest_partial[1] != -1:
            self.walkers = list(seen[longest_partial[0]])
            return longest_partial[1]

        raise TokenRejected("No valid token found")

    def generate_logit_bias_mask(self, logits):
        """
        Masks invalid tokens in logits based on the current state of the acceptor. Returns a bias.

        Args:
            logits: The logits tensor to mask. Just used for dimensionality.

        Returns:
            The bias, of the same type as `logits`.
        """
        valid_prefixes = set()
        for walker in self.walkers:
            valid_prefixes.update(walker.find_valid_prefixes(self.dawg))

        if not valid_prefixes:
            return get_logit_bias(logits, set())

        token_ids = [
            token_id
            for prefix in valid_prefixes
            if (token_id := self.vocabulary.get(prefix)) is not None
        ]

        return get_logit_bias(logits, set(token_ids))

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

            # Get walkers that accept this exact token
            new_walkers = [
                walker
                for valid_token, walker in StateMachine.advance_all(
                    self.walkers, token, self.dawg
                )
                if valid_token == token
            ]

            # Update walkers if we found valid transitions
            if new_walkers:
                self.walkers = new_walkers

    # -------- Private Methods --------

    def _waiting_for_trigger(self) -> bool:
        """
        Determines if the acceptor is waiting for the opening delimiter.

        Returns:
            bool: True if waiting for trigger, False otherwise.
        """
        if not self.acceptor or not isinstance(self.acceptor, EncapsulatedAcceptor):
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
