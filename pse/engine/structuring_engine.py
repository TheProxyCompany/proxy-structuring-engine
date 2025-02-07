from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pse_core.engine import Engine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.engine import StructuringMachine
from pse.grammar.python import PythonGrammar
from pse.json import JSONSchemaSource
from pse.util.get_top_logits import get_top_logits

logger = logging.getLogger(__name__)

Array_Type = TypeVar("Array_Type")
OutputType = TypeVar("OutputType")

class StructuringEngine(Engine):
    """
    The types of objects that the engine can use as a schema.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase,
    ) -> None:
        """
        Initialize the StructuringEngine with a tokenizer and vocabulary.
        """
        self.tokenizer = tokenizer
        reverse_vocab: dict[int, str] = {}
        for token, token_id in self.tokenizer.get_vocab().items():
            if "▁" == token:
                token = " "
            else:
                token = self.tokenizer.decode(token_id)
            reverse_vocab[token_id] = token
        super().__init__(reverse_vocab)

    def process_logits(self, _: Any, raw_logits: Array_Type) -> Array_Type:
        """
        Process the logits and return the processed logits.
        """
        self.multi_token_mapping: dict[int, list[int]] = {}
        tic = time.perf_counter()
        logger.debug(self._print_top_logits(raw_logits, 5, "🔵 Before processing"))
        adjusted_logits = super().process_logits(raw_logits)
        logger.debug(self._print_top_logits(adjusted_logits, 5, "🟢 After processing"))
        toc = time.perf_counter()
        logger.debug(f"Logit processing took {toc - tic:0.4f} seconds")
        return adjusted_logits

    def sample(
        self, logprobs: Array_Type, sampler: Callable[..., Array_Type], **kwargs: Any
    ) -> Array_Type:
        """
        Sample a token from the logits using the given sampler.
        kwargs are passed to the sampler function.
        """
        logger.debug(f"Sampling with kwargs: {kwargs}")
        tic = time.perf_counter()
        token = super().sample(logprobs, sampler)
        toc = time.perf_counter()
        logger.debug(f"Sampling took {toc - tic:0.4f} seconds: \033[33m{token}\033[0m")
        return type(logprobs)(token)  # type: ignore[reportCallIssue]

    def configure(self, schema: JSONSchemaSource, **kwargs: Any) -> None:
        """
        Configure the structuring engine with a schema and optional delimiters.

        Args:
            schema: Schema to use when structuring output
        """
        self.state_machine: StructuringMachine | None = StructuringMachine(
            schema, **kwargs
        )
        self.steppers = self.state_machine.get_steppers() if self.state_machine else []

    def get_current_state(self) -> str:
        non_accepted_state = ""
        for stepper in self.steppers:
            if stepper.has_reached_accept_state():
                return str(stepper.current_state)
            else:
                non_accepted_state = str(stepper.current_state)

        return non_accepted_state

    def parse_structured_output(
        self,
        raw_output: str | None = None,
        output_type: type[OutputType] | None = None,
    ) -> OutputType | Any:
        """
        Parse and cast the output to the given type.

        Args:
            raw_output: The raw string output to parse. If None, attempts to get from steppers.
            output_type: The type to cast the output to. If None, returns parsed but uncast value.

        Returns:
            Parsed and optionally cast output value.

        Raises:
            ValueError: If parsing fails in an unexpected way
            TypeError: If output type conversion fails
        """
        # Get output from steppers if none provided
        if not raw_output and self.steppers:
            for stepper in self.steppers:
                if stepper.has_reached_accept_state():
                    raw_output = stepper.get_current_value()
                    break

        if not raw_output:
            return None

        current_state = self.get_current_state()
        # Handle JSON parsing
        if current_state == "json" and isinstance(raw_output, str):
            if self.state_machine and self.state_machine.json_delimiters:
                raw_output = self._extract_between_delimiters(
                    raw_output,
                    self.state_machine.json_delimiters
                )
            try:
                raw_output = json.loads(raw_output)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                pass
        elif current_state == "python":
            raw_output = self._extract_between_delimiters(
                raw_output,
                PythonGrammar.delimiters
            )
            return raw_output

        # Handle type conversion if needed
        if (
            output_type
            and issubclass(output_type, BaseModel)
            and isinstance(raw_output, dict)
        ):
            try:
                return output_type.model_validate(raw_output)
            except Exception as e:
                logger.error(f"Failed to convert to {output_type}: {e}")
                pass

        return raw_output

    def _extract_between_delimiters(
        self, text: str, delimiters: tuple[str, str]
    ) -> str:
        """Extract content between start and end delimiters."""
        start, end = delimiters

        if start in text:
            text = text.split(start, 1)[1]
        if end in text:
            text = text.split(end, 1)[0]

        return text

    def _print_top_logits(self, logits: Any, top_n: int = 10, flag: str = "🔵") -> str:
        """
        Print the top logits for the given input and scores.
        """
        if logger.getEffectiveLevel() > logging.DEBUG:
            return ""

        rows = []
        top_logits = get_top_logits(logits, top_n)
        for token_id, score in top_logits.items():
            # check if score is too low to be considered
            if score == float("-inf") or score < -1e10:
                continue
            token = repr(self.tokenizer.decode(token_id))
            if token_id in self.multi_token_mapping:
                multiple_token_ids = self.multi_token_mapping[token_id]
                representation = repr(self.tokenizer.decode(multiple_token_ids))
                token = f"{token} -📶-> {representation}"

            rows.append(f"{token_id:<8} | {score:>10.4f} | {token}")

        header = f"{'Token ID':<8} | {'Score':>10} | Token"
        separator = "-" * 9 + "+" + "-" * 12 + "+" + "-" * 20
        chart = "\n".join([header, separator] + rows[:top_n])
        if rows:
            return f"{flag}\n{chart}"
        else:
            return f"{flag} No valid tokens found"
