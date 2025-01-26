from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

from pse_core.engine import Engine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.state_machines import get_state_machine
from pse.state_machines.composite.encapsulated import EncapsulatedStateMachine
from pse.state_machines.composite.wait_for import WaitFor
from pse.util.generate_schema import callable_to_json_schema, pydantic_to_json_schema
from pse.util.get_top_logits import get_top_logits

logger = logging.getLogger(__name__)

LogitType = TypeVar("LogitType")
OutputType = TypeVar("OutputType", bound=BaseModel)


@dataclass
class EngineOutput[T]:
    """
    A dataclass that represents the output of the Engine.
    """

    """
    The buffer is input that was not used in the state machine. For example, using a schema with delimiters,
    the buffer is the input before the first delimiter.
    """
    buffer: str
    """
    The value output by the engine.
    """
    value: T


class StructuringEngine(Engine):

    StructureType: TypeAlias = type[BaseModel] | list[type[BaseModel]] | dict[str, Any] | list[dict[str, Any]] | Callable[..., Any]
    """
    The types of objects that the engine can use as a schema.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase) -> None:
        """
        Initialize the StructuringEngine with a tokenizer and vocabulary.
        """
        self.tokenizer = tokenizer
        raw_vocab = self.tokenizer.get_vocab()
        reverse_vocab: dict[int, str] = {}
        for token, token_id in raw_vocab.items():
            if "▁" == token:
                token = " "
            else:
                token = self.tokenizer.decode(token_id)
            reverse_vocab[token_id] = token
        super().__init__(reverse_vocab)

    def encode_token(self, token: str) -> list[int]:
        """
        Encode a token into a list of token IDs.
        This helper method allows us to avoid passing the tokenizer object to the C++ engine.
        """
        return self.tokenizer.encode(token, add_special_tokens=False)

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
        self.print_logits(scores, 10, "🔵 Before")
        tic = time.perf_counter()
        adjusted_logits = super().process_logits(scores)
        toc = time.perf_counter()
        self.print_logits(adjusted_logits, 10, "🟢 After")
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
        decoded_token = self.tokenizer.decode(token)
        logger.debug(f"Sampled token: {decoded_token!r}")
        return token

    def configure(
        self,
        schema: StructureType | None = None,
        delimiters: tuple[str, str] | None = None,
        buffer_length: int = -1,
    ) -> None:
        """
        Configure the structuring engine with a schema and optional delimiters.

        Args:
            schema: Schema to validate structured output against
            delimiters: Optional tuple of (start, end) delimiters that indicate the start and end of the structured output
            buffer_length: Controls when schema validation begins. Can be used with or without delimiters. Defaults to -1.

        Note:
            - buffer_length == -1: Optional buffer with no minimum length (default)
            - buffer_length == 0: Immediate schema validation. No buffer is allowed.
            - buffer_length > 0: Buffer must reach specified length before validation
            - If delimiters are provided, the buffer length is ignored.
        """
        if schema is None:
            return

        self.schema = self.get_schema_object(schema)
        self.state_machine = get_state_machine(self.schema)

        if delimiters:
            self.state_machine = EncapsulatedStateMachine(
                self.state_machine,
                delimiters,
                buffer_length,
            )
        elif buffer_length != 0:
            self.state_machine = WaitFor(self.state_machine, buffer_length)

        self.steppers = self.state_machine.get_steppers()

    def reset(self) -> None:
        """
        Reset the state machine and steppers.
        """
        self.steppers = self.state_machine.get_steppers()

    def print_logits(self, scores: Any, top_n: int = 64, flag: str = "🔵") -> None:
        """
        Print the top logits for the given input and scores.
        """
        if logger.getEffectiveLevel() > logging.DEBUG:
            return

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
            logger.debug(f"{flag} Top logits:\n{chart}")
        else:
            logger.debug(f"{flag} No valid tokens found")

    def read_output(self, output_type: type[OutputType] | type[Any] | None = None) -> Iterable[EngineOutput[Any]] | Iterable[EngineOutput[OutputType]]:
        """
        Get the current value of the structuring engine.
        """
        for stepper in self.steppers:
            buffer = ""
            value = None
            stepper_value = stepper.get_current_value()
            if isinstance(stepper_value, tuple):
                buffer = stepper_value[0]
                value = stepper_value[1]
            else:
                value = stepper_value

            if output_type and value is not None:
                try:
                    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                        # Handle Pydantic models
                        value = output_type.model_validate(value)
                    elif type(value) is not output_type:
                        # Handle primitive types
                        value = output_type(value)
                    yield EngineOutput[output_type](buffer, value)
                except Exception:
                    breakpoint()
                    logger.warning(
                        f"Failed to validate/cast value {value} with type {output_type}"
                    )

            yield EngineOutput[type(value)](buffer, value)


    @staticmethod
    def get_schema_object(schema: StructureType) -> dict[str, Any]:
        """
        Convert the given schema into an object that can be used by the engine.
        """
        if isinstance(schema, list):
            if all(isinstance(s, type) and issubclass(s, BaseModel) for s in schema):
                return {
                    "oneOf": [
                        pydantic_to_json_schema(s)
                        for s in schema
                        if isinstance(s, type) and issubclass(s, BaseModel)
                    ]
                }
            else:
                return {"oneOf": schema}
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            return pydantic_to_json_schema(schema)
        elif callable(schema):
            return callable_to_json_schema(schema)
        elif isinstance(schema, dict):
            if "schema" in schema:
                return schema["schema"]
            else:
                return schema
