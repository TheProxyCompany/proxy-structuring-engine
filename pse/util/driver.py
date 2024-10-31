from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from lexpy import DAWG
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.generation.logits_process import LogitsProcessor

from pse.acceptors.collections.encapsulated_acceptor import EncapsulatedAcceptor
from pse.acceptors.token_acceptor import TokenAcceptor
from pse.util.errors import TokenRejected
from pse.util.get_acceptor import get_json_acceptor
from pse.state_machine.walker import Walker
from pse.state_machine.state_machine import StateMachine
from pse.util.bias_logits import bias_logits
from pse.util.handle_logits import handle_logits

logger = logging.getLogger(__name__)


class OutputType(Enum):
    """
    Represents the type of structured schema constraints for output.

    Attributes:
        JSON (OutputType): Output constrained by a JSON schema.
        PYTHON (OutputType): Output constrained by a Python schema.
        CYPHER (OutputType): Output constrained by a Cypher schema.
        U_DIFF (OutputType): Output constrained by a U_DIFF schema.
    """

    JSON = (("```json\n", "\n```"),)
    PYTHON = (("```python\n", "\n```"),)
    CYPHER = (("```cypher\n", "\n```"),)
    U_DIFF = (("```diff\n", "\n```"),)

    def __init__(self, delimiters: Tuple[str, str]) -> None:
        self._delimiters = delimiters

    @property
    def delimiters(self) -> Tuple[str, str]:
        """
        Returns the opening and closing delimiters based on the OutputType.

        Returns:
            Tuple[str, str]: A tuple containing the opening and closing delimiters.
        """
        return self._delimiters


class StructuredOutputDriver(LogitsProcessor):
    """
    Drives a StateMachineAcceptor to manage and validate structured outputs based on a given schema.

    This driver utilizes various acceptors to ensure that the output adheres to the specified JSON schema
    or other supported schema types. It manages the state of token acceptance and provides mechanisms
    to advance tokens and characters while validating the structured output.
    """

    def __init__(
        self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    ) -> None:
        """
        Initializes the StructuredOutputDriver with the provided tokenizer.

        Args:
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer used to convert tokens to strings.
        """
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = tokenizer
        self.eos_id: int = tokenizer.eos_token_id or 0
        self.acceptor: Optional[TokenAcceptor] = None
        self.walkers: List[Walker] = []
        self.within_json_value: bool = False
        self._build_vocabulary()

    @property
    def can_accept_input(self) -> bool:
        """
        Checks if the acceptor can accept input.

        Returns:
            bool: True if the acceptor can accept input, False otherwise.
        """
        return self.acceptor is not None and not self.has_reached_accept_state

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
        if self._waiting_for_trigger():
            return False

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

    def reset(self) -> None:
        """
        Resets the driver to its initial state.
        """
        self.walkers = list(self.acceptor.get_walkers()) if self.acceptor else []
        self.within_json_value = False

    def create_acceptor(
        self,
        schema: Optional[Dict[str, Any]] = None,
        output_type: OutputType = OutputType.JSON,
        encapsulated: bool = True,
        delimiters: Optional[Tuple[str, str]] = None,
    ) -> None:
        """
        Creates a StateMachineAcceptor based on the output types and provided schema.

        This method initializes the appropriate acceptor(s) for each specified OutputType.
        If encapsulation is enabled, it wraps the acceptor within the defined delimiters.

        Args:
            schema (Optional[Dict[str, Any]]): The JSON schema used to validate or constrain the output.
            output_type (OutputType): The type of structured output.
            encapsulated (bool): Whether to encapsulate the acceptor with delimiters.
            delimiters (Optional[Tuple[str, str]]): Custom delimiters.

        Raises:
            ValueError: If an unsupported output type is provided.
        """
        if not schema:
            self.acceptor = None
            self.walkers = []
            return

        if output_type == OutputType.JSON:
            acceptor = get_json_acceptor(
                schema,
                start_hook=self._start_hook,
                end_hook=self._end_hook,
            )
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        if encapsulated:
            if delimiters is None:
                delimiters = output_type.delimiters

            acceptor = EncapsulatedAcceptor(
                acceptor,
                delimiters[0],
                delimiters[1],
            )

        self.acceptor = acceptor
        self.walkers = list(self.acceptor.get_walkers()) if self.acceptor else []

    def mask_invalid_tokens(self, logits):
        """
        Masks invalid tokens in logits based on the current state of the acceptor.

        Args:
            logits: The logits tensor to be modified.

        Returns:
            Modified logits with invalid tokens masked.
        """
        valid_prefixes: Set[str] = set()
        for walker in self.walkers:
            valid_prefixes.update(walker.find_valid_prefixes(self.dawg))

        token_ids: List[int] = [
            token_id
            for prefix in valid_prefixes
            if (token_id := self.token_to_id.get(prefix)) is not None
        ]

        return bias_logits(logits, set(token_ids))

    def get_valid_token(self, logits, num_top_tokens: int = 8) -> int:
        """
        Advances the acceptor's state using the provided logits.

        Args:
            logits: The logits from the language model.
            num_top_tokens (int): The number of top tokens to consider.

        Returns:
            int: The token ID of the next token to generate.

        Raises:
            TokenRejected: If no valid token is found.
        """
        top_tokens = self._get_top_tokens(logits, num_top_tokens)


        for index, (token_id, token, score) in enumerate(top_tokens):
            logger.debug(
                f"{index + 1}. token_id: {token_id}, token: {token}, score: {score}"
            )
            full_match_walkers: List[Walker] = []
            for valid_token, walker in StateMachine.advance_all_walkers(
                self.walkers, token, self.dawg
            ):
                if valid_token != token:
                    logger.debug(f"{index + 1}. partial match: {valid_token}")
                    logger.warning(f"Partial matches not implemented: {valid_token}")
                    continue
                else:
                    logger.debug(f"{index + 1}. valid_token: {valid_token}")
                    full_match_walkers.append(walker)

                if walker.has_reached_accept_state():
                    self.walkers = full_match_walkers
                    return token_id

            if full_match_walkers:
                self.walkers = full_match_walkers
                return token_id

        raise TokenRejected(f"No valid token found in the top {num_top_tokens} tokens")

    # -------- Private Methods --------

    def _build_vocabulary(self) -> None:
        """
        Integrates the tokenizer's vocabulary into the driver.
        Maintains both the original and decoded forms of tokens.
        """
        self.dawg = DAWG()
        self.special_tokens_set = set()
        self.token_to_id: Dict[str, int] = {}  # mapping of decoded token to its ID
        token_ids = list(self.tokenizer.get_vocab().values())
        decoded_tokens: List[str] = self.tokenizer.batch_decode(token_ids)

        for decoded_token, token_id in sorted(zip(decoded_tokens, token_ids)):
            self.dawg.add(decoded_token)
            if decoded_token in self.tokenizer.all_special_tokens:
                self.special_tokens_set.add(decoded_token)

            self.token_to_id[decoded_token] = token_id

        self.vocabulary = set(self.token_to_id.keys()) - self.special_tokens_set
        self.dawg.reduce()

    def _get_top_tokens(
        self, logits, num_tokens: int = 64
    ) -> List[Tuple[int, str, float]]:
        """
        Returns the top tokens from the logits.

        Args:
            logits: The logits tensor from the language model.
            num_tokens (int): The number of top tokens to consider.

        Returns:
            List[Tuple[int, str, float]]: A list of tuples containing the token ID, token, and raw score.
        """
        top_logits = handle_logits(logits, num_tokens)
        top_logits.sort(key=lambda x: x[1], reverse=True)
        token_ids = [token_id for token_id, _ in top_logits]
        tokens = self.tokenizer.batch_decode(token_ids)
        top_tokens = list(zip(token_ids, tokens, [score for _, score in top_logits]))
        return top_tokens

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
