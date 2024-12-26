from __future__ import annotations

import logging
import pprint
from typing import Any

from pse_core.engine import Engine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.state_machines.collections.encapsulated_acceptor import EncapsulatedAcceptor
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

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase,
        vocabulary: dict[str, int] | None = None,
    ) -> None:
        """
        Initialize the StructuringEngine with a tokenizer and vocabulary.
        """
        self.tokenizer = tokenizer
        raw_vocab = vocabulary or self.tokenizer.get_vocab()
        token_ids = list(raw_vocab.values())
        decoded_tokens = (
            list(raw_vocab.keys())
            if vocabulary
            else self.tokenizer.batch_decode(token_ids)
        )
        reverse_vocab = {id: token for token, id in zip(decoded_tokens, token_ids, strict=True)}
        super().__init__(reverse_vocab)

    def encode_token(self, token: str) -> list[int]:
        """
        Encode a token into a list of token IDs.
        This helper method allows us to avoid passing the tokenizer object to the C++ engine.
        """
        return self.tokenizer.encode(token, add_special_tokens=False)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """
        Process the logits and return the next token.
        Invokes the C++ engine to process the logits.
        """
        adjusted_logits = self.process_logits(input_ids, scores)
        # self.print_logits(adjusted_logits, 10)
        return adjusted_logits

    def configure(
        self,
        schema: type[BaseModel]
        | list[type[BaseModel]]
        | dict[str, Any]
        | list[dict[str, Any]],
        wrap_with_delimiters: bool = False,
        delimiters: tuple[str, str] | None = ("```json\n", "\n```"),
    ) -> None:
        """
        Configure the structuring engine with the given schema.
        """

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

        state_machine = get_state_machine(self.schema)
        self.state_machine = (
            state_machine
            if not self.is_encapsulated
            else EncapsulatedAcceptor(state_machine, self.delimiters)
        )
        self.walkers = self.state_machine.get_walkers()

    def __repr__(self) -> str:
        return (
            "StructuringEngine(\n"
            f"    {len(self.walkers)} walkers\n"
            f"    schema={pprint.pformat(self.schema, indent=4, width=80)}\n"
            ")"
        )

    def reset(self) -> None:
        """
        Reset the state machine and walkers.
        """
        self.walkers = self.state_machine.get_walkers()

    def print_logits(self, scores: Any, top_n: int = 64) -> None:
        """
        Print the top logits for the given input and scores.
        """

        rows = []
        top_logits = get_top_logits(scores, top_n)

        for token_id, score in top_logits.items():
            # Get token from token_id using reverse vocabulary map
            if not (token := self.reverse_vocabulary.get(token_id)):
                logger.warning(f"Unknown token ID: {token_id}")
                continue

            rows.append(f"{token_id:<8} | {score:>10.4f} | {repr(token)[1:-1]}")

        header = f"{'Token ID':<8} | {'Score':>10} | Token"
        separator = "-" * 9 + "+" + "-" * 12 + "+" + "-" * 20
        chart = "\n".join([header, separator] + rows[:top_n])
        if rows:
            logger.debug(f"🔵 Top logits:\n{chart}")
        else:
            logger.debug("🔴 No valid tokens found")
