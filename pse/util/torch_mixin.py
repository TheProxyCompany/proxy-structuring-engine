import os
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from transformers import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.cache_utils import StaticCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.utils import ModelOutput

from pse.engine.structuring_engine import StructuringEngine


class PSETorchMixin(GenerationMixin):
    engine: StructuringEngine

    @staticmethod
    def make_sampler(do_sample: bool) -> Callable[[torch.Tensor], torch.Tensor]:
        def sampler(x: torch.Tensor) -> torch.Tensor:
            probs = nn.functional.softmax(x, dim=-1)
            if torch.isinf(probs).any() or torch.isnan(probs).any():
                raise ValueError("All logits were masked.")
            if do_sample:
                return torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                return torch.argmax(probs, dim=-1)

        return sampler

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: BaseStreamer | None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample
        if self.engine.process_logits not in logits_processor:
            # insert the engine at the beginning of the list
            logits_processor.insert(0, self.engine.process_logits)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished: bool = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__  # type: ignore [attr-defined]
        if isinstance(model_kwargs.get("past_key_values"), StaticCache):
            if self.device.type == "cuda" and self.get_compiled_call:  # type: ignore [attr-defined]
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)  # type: ignore [attr-defined]

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            if is_prefill:
                assert isinstance(self, Callable)
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            sampler = PSETorchMixin.make_sampler(do_sample)
            next_tokens = self.engine.sample(next_token_scores, sampler).long()
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            # update generated ids, model inputs, and length for next step
            if len(next_tokens) > 1:
                input_ids = torch.cat([input_ids, next_tokens[None]], dim=-1)  # type: ignore[arg-type]
            elif next_tokens.shape[0] > 0:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)  # type: ignore[arg-type]
            else:
                break

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,  # type: ignore [attr-defined]
                num_new_tokens=len(next_tokens),
            )
            if synced_gpus and this_peer_finished:
                continue

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    assert scores is not None
                    scores += (next_token_scores,)
                if output_logits:
                    assert raw_logits is not None
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    assert decoder_attentions is not None
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder  # type: ignore [attr-defined]
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:  # type: ignore [attr-defined]
                        assert cross_attentions is not None
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    assert decoder_hidden_states is not None
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder  # type: ignore [attr-defined]
                        else (outputs.hidden_states,)
                    )
            if streamer is not None:
                streamer.put(next_tokens[None].cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = bool(unfinished_sequences.max() == 0)
            cur_len += len(next_tokens)  # count new tokens
            del outputs
            if self.engine.has_reached_accept_state:
                break

        if streamer is not None:
            streamer.end()

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        def _extract_past_from_model_output(outputs: ModelOutput):
            past_key_values = None
            cache_name = "past_key_values"
            if "past_key_values" in outputs:
                past_key_values = outputs.past_key_values  # type: ignore [attr-defined]
            elif "mems" in outputs:
                past_key_values = outputs.mems  # type: ignore [attr-defined]
            elif "past_buckets_states" in outputs:
                past_key_values = outputs.past_buckets_states  # type: ignore [attr-defined]
            elif "cache_params" in outputs:
                past_key_values = outputs.cache_params  # type: ignore [attr-defined]
                cache_name = "cache_params"

            return cache_name, past_key_values

        # update past_key_values keeping its naming used in model code
        cache_name, cache = _extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state  # type: ignore [attr-defined]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (attention_mask.shape[0], num_new_tokens)
                        ),
                    ],
                    dim=-1,
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], num_new_tokens)
                        ),
                    ],
                    dim=-1,
                )

        past_positions = model_kwargs.pop("cache_position")
        new_positions = torch.arange(
            past_positions[-1] + 1,
            past_positions[-1] + 1 + num_new_tokens,
            dtype=past_positions.dtype,
        ).to(past_positions.device)

        if not model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        else:
            model_kwargs["cache_position"] = new_positions

        return model_kwargs
