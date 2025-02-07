from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pse_core.engine import Engine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.json import JSONSchemaSource
from pse.state_machine import build_state_machine
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

    def __call__(self, _: Any, raw_logits: Array_Type) -> Array_Type:
        """
        Logit Processing api: (input, logits) -> new_logits

        Args:
            raw_logits: The logits to process.

        Returns:
            The processed logits.
        """
        return self.process_logits(None, raw_logits)

    def process_logits(self, _: Any, raw_logits: Array_Type) -> Array_Type:
        """
        Process the logits and return the processed logits.
        """
        self.multi_token_mapping: dict[int, list[int]] = {}
        tic = time.perf_counter()
        logger.debug(self.chart_model_output(raw_logits, 5, "🔵 Before processing"))
        adjusted_logits = super().process_logits(raw_logits)
        logger.debug(self.chart_model_output(adjusted_logits, 5, "🟢 After processing"))
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
        self.state_machine = build_state_machine(schema, **kwargs)
        self.steppers = self.state_machine.get_steppers()

    def cast_output(
        self, output: Any | None = None, output_type: type[OutputType] | Any = Any
    ) -> Any:
        """
        Cast the output to the given type.

        Args:
            output_type: The type of the output to return. If None, return the raw values.
        """
        # if no input to cast, find accepted stepper and use its value
        if output is None and self.steppers:
            output = self.steppers[0].get_current_value()

        if not output:
            return None

        if isinstance(output, str):
            output = output.strip()

        # clean delimiters if present
        # if self.delimiters and isinstance(output, str):
        #     if output.startswith(self.delimiters[0]):
        #         output = output[len(self.delimiters[0]) :]

        #     if output.endswith(self.delimiters[1]):
        #         output = output[: -len(self.delimiters[1])]

        assert output is not None
        try:
            # cast to json if string
            value = json.loads(output) if isinstance(output, str) else output
            # validate with pydantic if BaseModel
            if output_type is not None and issubclass(output_type, BaseModel):
                value = output_type.model_validate(value)
            return value
        except Exception:
            breakpoint()
            logger.warning(f"Failed to cast value {output} with type {output_type}")
            return None

    def reset(self, hard: bool = False) -> None:
        """
        Reset the state machine and steppers.
        """
        if hard:
            self.state_machine = None
            self.steppers = []
        elif self.state_machine:
            self.steppers = self.state_machine.get_steppers()

    @property
    def in_accepted_state(self) -> bool:
        """
        Check if the state machine is in an accepted state.
        """
        return self.has_reached_accept_state

    def chart_model_output(self, scores: Any, top_n: int = 10, flag: str = "🔵") -> str:
        """
        Print the top logits for the given input and scores.
        """
        if logger.getEffectiveLevel() > logging.DEBUG:
            return ""

        rows = []
        top_logits = get_top_logits(scores, top_n)
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
