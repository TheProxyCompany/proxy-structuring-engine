from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pse_core.engine import Engine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.state_machines import build_state_machine
from pse.state_machines.composite.encapsulated import EncapsulatedStateMachine
from pse.state_machines.composite.wait_for import WaitFor
from pse.structure import SchemaType, get_schema
from pse.util.get_top_logits import get_top_logits

logger = logging.getLogger(__name__)

LogitType = TypeVar("LogitType")
OutputType = TypeVar("OutputType")

@dataclass
class EngineOutput[OutputType]:
    """
    The value output by the engine.
    """

    value: OutputType | Any

    """
    The buffer is input that was not used by the state machine.
    """
    buffer: str


class StructuringEngine(Engine):
    """
    The types of objects that the engine can use as a schema.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase
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

    def __call__(self, scores: LogitType) -> LogitType:
        """
        Merge invalid token scores with the valid token scores.
        i.e
            Hi: 10.0
            "Hi: 5.0
            _______
            Hi: -inf
            "Hi: 15.0
        """
        return self.process_logits(scores)

    def process_logits(self, scores: LogitType) -> LogitType:
        self.multi_token_mapping: dict[int, list[int]] = {}
        tic = time.perf_counter()
        logger.debug(self.chart_model_output(scores, 3, "🔵 Before processing"))
        adjusted_logits = super().process_logits(scores)
        logger.debug(self.chart_model_output(adjusted_logits, 3, "🟢 After processing"))
        toc = time.perf_counter()
        logger.debug(f"Logit processing took {toc - tic:0.4f} seconds")
        return adjusted_logits

    def sample(
        self, logprobs: object, sampler: Callable[..., object], **kwargs
    ) -> list[int]:
        """
        Sample a token from the logits using the given sampler.
        kwargs are passed to the sampler function.
        """
        logger.debug(f"Sampling with kwargs: {kwargs}")
        tic = time.perf_counter()
        token = super().sample(logprobs, sampler, **kwargs)
        toc = time.perf_counter()
        logger.debug(f"Sampling took {toc - tic:0.4f} seconds")
        return token

    def configure(
        self,
        schema: SchemaType,
        delimiters: tuple[str, str] | None = None,
        buffer_length: int = -1,
    ) -> None:
        """
        Configure the structuring engine with a schema and optional delimiters.

        Args:
            schema: Schema to use when structuring output
            delimiters:
                Tuple (start, end) delimiters that indicate the start and end of the structured output.
                Defaults to None.
            buffer_length:
                Controls when schema validation begins. Can be used with or without delimiters. Defaults to -1.
        Note:
            - buffer_length == -1: Optional buffer with no minimum length (default)
            - buffer_length == 0: Immediate schema validation. No buffer is allowed.
            - buffer_length > 0: Buffer must reach specified length before validation
            - If delimiters are provided, the buffer length is ignored.
        """

        self.schema = get_schema(schema)
        self.state_machine = build_state_machine(self.schema)

        if delimiters:
            self.state_machine = EncapsulatedStateMachine(
                self.state_machine,
                delimiters,
                buffer_length,
            )
        elif buffer_length != 0:
            self.state_machine = WaitFor(self.state_machine, buffer_length)

        self.steppers = self.state_machine.get_steppers()

    def output(self, output_type: type[OutputType] | Any = Any) -> EngineOutput[OutputType]:
        """
        Get the current values of the structuring engine.

        Args:
            output_type: The type of the output to return. If None, return the raw values.
        """

        buffer: str = ""
        value: OutputType | Any = None
        if not self.steppers:
            return EngineOutput[OutputType](value, buffer)

        for stepper in self.steppers:
            stepper_value = stepper.get_current_value()
            if isinstance(stepper_value, tuple):
                buffer = stepper_value[0]
                value = stepper_value[1]
            else:
                value = stepper_value

            if not value:
                value = buffer
            elif output_type is not None and issubclass(output_type, BaseModel):
                try:
                    value = output_type.model_validate(value)
                except Exception:
                    logger.warning(f"Failed to cast value {value} with type {output_type}")

        return EngineOutput[OutputType](value, buffer)

    def reset(self) -> None:
        """
        Reset the state machine and steppers.
        """
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
            # Get token from token_id using reverse vocabulary map
            if not (token := self.reverse_vocabulary.get(token_id)):
                logger.warning(f"Unknown token ID: {token_id}")
                continue

            if score == float("-inf"):
                continue

            if token_id in self.multi_token_mapping:
                multiple_token_ids = self.multi_token_mapping[token_id]
                token = repr(self.tokenizer.decode(multiple_token_ids)) + " *️⃣"
            else:
                token = repr(token)

            rows.append(f"{token_id:<8} | {score:>10.4f} | {token}")

        header = f"{'Token ID':<8} | {'Score':>10} | Token"
        separator = "-" * 9 + "+" + "-" * 12 + "+" + "-" * 20
        chart = "\n".join([header, separator] + rows[:top_n])
        if rows:
            return f"{flag}\n{chart}"
        else:
            return f"{flag} No valid tokens found"
