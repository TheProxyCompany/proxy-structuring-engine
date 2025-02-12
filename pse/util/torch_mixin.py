import os
from collections.abc import Callable

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

from pse.engine.structuring_engine import StructuringEngine


class PSETorchMixin(GenerationMixin):
    engine: StructuringEngine

    @staticmethod
    def make_sampler(do_sample: bool) -> Callable:
        if do_sample:
            return lambda x: torch.multinomial(x, num_samples=1).squeeze(1)
        else:
            return lambda x: torch.argmax(x, dim=-1)

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
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
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

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,  # type: ignore [attr-defined]
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
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

            # token selection
            log_probs = nn.functional.softmax(next_token_scores, dim=-1)
            sampler = PSETorchMixin.make_sampler(do_sample)
            next_tokens = self.engine.sample(log_probs, sampler).long()
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat(
                [input_ids, next_tokens[:, None]], # type: ignore[arg-type]
                dim=-1
            )
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = bool(unfinished_sequences.max() == 0)
            cur_len += len(next_tokens) # count new tokens
            del outputs
            if self.engine.has_reached_accept_state:
                break

        if streamer is not None:
            streamer.end()

        return input_ids
