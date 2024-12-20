from __future__ import annotations

import logging
from typing import Any

from pse_core.engine import Engine
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from pse.state_machines.collections.encapsulated_acceptor import EncapsulatedAcceptor
from pse.state_machines.get_state_machine import get_state_machine

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
        self.tokenizer = tokenizer
        raw_vocab = vocabulary or self.tokenizer.get_vocab()
        token_ids = list(raw_vocab.values())
        decoded_tokens = (
            list(raw_vocab.keys())
            if vocabulary
            else self.tokenizer.batch_decode(token_ids)
        )
        vocab = {token: id for token, id in zip(decoded_tokens, token_ids, strict=True)}
        super().__init__(vocab)

    def encode_token(self, token: str) -> list[int]:
        return self.tokenizer.encode(token, add_special_tokens=False)

    # def __call__(self, input_ids: Any, scores: Any) -> Any:
    #     """
    #     scores are logits
    #     """
    #     # Generate rows for each token
    #     rows = []
    #     top_logits = get_top_logits(scores, 20)
    #     valid_tokens, reversed_valid_tokens = self.get_valid_tokens()
    #     if not valid_tokens:
    #         return scores

    #     for token_id, score in top_logits.items():
    #         # Get token from token_id using reverse vocabulary map
    #         if not (token := self.reverse_vocabulary.get(token_id)):
    #             logger.warning(f"Unknown token ID: {token_id}")
    #             continue

    #         if token in valid_tokens:
    #             rows.append(f"{token_id:<8} | 🟢 {score:>9.4f} | {repr(token)[1:-1]}")
    #             continue

    #         if token not in valid_tokens:
    #             scores[token_id] = float("-inf")
    #             fixed_tokens = [
    #                 t[::-1]
    #                 for t in reversed_valid_tokens.search_with_prefix(token[::-1])
    #                 if isinstance(t, str)
    #             ]

    #             for fixed_token in fixed_tokens:
    #                 fixed_token_id = self.vocabulary[fixed_token]
    #                 scores[fixed_token_id] = score
    #                 rows.append(
    #                     f"{fixed_token_id:<8} | 🟢 {score:>9.4f} | {repr(fixed_token)[1:-1]}"
    #                 )

    #     header = f"{'Token ID':<8} | {'Score':>10} | Token"
    #     separator = "-" * 9 + "+" + "-" * 12 + "+" + "-" * 20
    #     chart = "\n".join([header, separator] + rows[:10])
    #     if rows:
    #         logger.info(f"🔵 Top logits:\n{chart}")
    #     else:
    #         logger.info("🔴 No valid tokens found")

    #     valid_token_ids = set(self.vocabulary[t] for t in valid_tokens)
    #     return scores + get_logit_bias(scores, valid_token_ids)

    def configure(
        self,
        schema: type[BaseModel]
        | list[type[BaseModel]]
        | dict[str, Any]
        | list[dict[str, Any]],
        wrap_with_delimiters: bool = False,
        delimiters: tuple[str, str] | None = ("```json\n", "\n```"),
    ) -> None:

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
