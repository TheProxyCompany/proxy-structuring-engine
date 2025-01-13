from __future__ import annotations

import logging
import pprint
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pse_core.engine import Engine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.state_machines.composite.encapsulated import EncapsulatedStateMachine
from pse.state_machines.composite.wait_for import WaitForStateMachine
from pse.state_machines.get_state_machine import get_state_machine
from pse.util.get_top_logits import get_top_logits

logger = logging.getLogger(__name__)


class StructuringEngine(Engine):
    """
    Drives a StateMachineAcceptor to manage and validate structured outputs based on a given schema.

    This driver utilizes various acceptors to ensure that the output adheres to the specified JSON schema
    or other supported schema types. It manages the state of token acceptance and provides mechanisms
    to advance tokens and characters while validating the structured output.
    """

    T = TypeVar("T")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase,
    ) -> None:
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

    def __call__(self, scores: T) -> T:
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

    def process_logits(self, scores: T) -> T:
        self.multi_token_mapping = {}
        self.print_logits(scores, 10, "🔵 Before")
        tic = time.perf_counter()
        adjusted_logits = super().process_logits(scores)
        toc = time.perf_counter()
        self.print_logits(adjusted_logits, 10, "🟢 After")
        logger.debug(f"Logit processing took {toc - tic:0.4f} seconds")
        return adjusted_logits

    def sample(
        self,
        logprobs: object,
        sampler: Callable[..., object],
        **kwargs,
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
        schema: type[BaseModel]
        | list[type[BaseModel]]
        | dict[str, Any]
        | list[dict[str, Any]],
        wrap_with_delimiters: bool = False,
        delimiters: tuple[str, str] | None = ("```json\n", "\n```"),
        wait_for_acceptor: bool = False,
    ) -> None:
        """
        Configure the structuring engine with the given schema.
        """
        if wait_for_acceptor and wrap_with_delimiters:
            raise ValueError("Cannot wait for acceptor and wrap with delimiters")

        self.is_encapsulated = wrap_with_delimiters
        if self.is_encapsulated:
            if not delimiters:
                raise ValueError(
                    "Delimiters must be provided if wrap_with_delimiters is True"
                )
            self.delimiters = delimiters

        if isinstance(schema, list):
            if all(isinstance(s, type) and issubclass(s, BaseModel) for s in schema):
                self.schema = {
                    "oneOf": [
                        s.model_json_schema()
                        for s in schema
                        if isinstance(s, type) and issubclass(s, BaseModel)
                    ]
                }
            else:
                self.schema = {"oneOf": schema}
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self.schema = schema.model_json_schema()
        elif isinstance(schema, dict):
            if "schema" in schema:
                logger.warning(
                    "Schema should not be provided as an object with 'schema' key."
                )
                self.schema = schema["schema"]
            else:
                self.schema = schema

        self.state_machine = get_state_machine(self.schema)
        if self.is_encapsulated:
            self.state_machine = EncapsulatedStateMachine(
                self.state_machine, self.delimiters
            )
        if wait_for_acceptor:
            self.state_machine = WaitForStateMachine(self.state_machine)

        self.walkers = self.state_machine.get_walkers()

    def __repr__(self) -> str:
        return (
            "StructuringEngine(\n"
            f"    schema={pprint.pformat(self.schema, indent=4, width=80)}\n"
            f"    walkers={self.walkers}\n"
            ")"
        )

    def reset(self) -> None:
        """
        Reset the state machine and walkers.
        """
        self.walkers = self.state_machine.get_walkers()

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
