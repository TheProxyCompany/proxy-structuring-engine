import inspect
from collections.abc import Callable

import tensorflow as tf
from transformers import TFGenerationMixin, TFLogitsProcessorList
from transformers.generation.tf_utils import (
    TFSampleDecoderOnlyOutput,
    TFSampleEncoderDecoderOutput,
    TFSampleOutput,
    shape_list,
)

from pse.engine.structuring_engine import StructuringEngine


class PSETFMixin(TFGenerationMixin):
    engine: StructuringEngine

    @staticmethod
    def make_sampler(num_samples: int, sample_seed: tuple[int, int]) -> Callable[[tf.Tensor], tf.Tensor]:
        return lambda x: tf.squeeze(
            tf.random.stateless_categorical(
                logits=x,
                num_samples=num_samples,
                seed=sample_seed,
                dtype=tf.int32,
            ),
            axis=1,
        )

    def sample(
        self,
        input_ids: tf.Tensor,
        logits_processor: TFLogitsProcessorList | None = None,
        logits_warper: TFLogitsProcessorList | None = None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        seed: tuple[int, int] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        return_dict_in_generate: bool | None = None,
        **model_kwargs,
    ) -> TFSampleOutput | tf.Tensor:
        r"""
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`TFLogitsProcessorList`, *optional*):
                An instance of [`TFLogitsProcessorList`]. List of instances of class derived from [`TFLogitsWarper`]
                used to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            seed (`List[int]`, *optional*):
                Random seed to control sampling, containing two integers, used when `do_sample` is `True`. See the
                `seed` argument from stateless functions in `tf.random`.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `call` function of the model. If model is an
                encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.TFSampleDecoderOnlyOutput`], [`~generation.TFSampleEncoderDecoderOutput`] or `tf.Tensor`: A
            `tf.Tensor` containing the generated tokens (default behaviour) or a
            [`~generation.TFSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.TFSampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     TFAutoModelForCausalLM,
        ...     TFLogitsProcessorList,
        ...     TFMinLengthLogitsProcessor,
        ...     TFTopKLogitsWarper,
        ...     TFTemperatureLogitsWarper,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="tf").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = TFLogitsProcessorList(
        ...     [
        ...         TFMinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = TFLogitsProcessorList(
        ...     [
        ...         TFTopKLogitsWarper(50),
        ...         TFTemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> tf.random.set_seed(0)
        >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and I love my country. But when I look at Donald Trump,']
        ```"""

        # 1. init greedy_search values
        logits_processor = (
            logits_processor
            if logits_processor is not None
            else TFLogitsProcessorList()
        )
        logits_warper = (
            logits_warper if logits_warper is not None else TFLogitsProcessorList()
        )
        if self.engine.process_logits not in logits_processor:
            # insert the engine at the beginning of the list
            logits_processor.insert(0, self.engine.process_logits)

        def pse_sample_fn(logits: tf.Tensor, sampler: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
            return self.engine.sample(logits, sampler)

        max_length = (
            max_length if max_length is not None else self.generation_config.max_length
        )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]  # type: ignore [assignment]
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        use_cache = model_kwargs.pop("use_cache", self.generation_config.use_cache)
        use_xla = not tf.executing_eagerly()
        # TODO (Joao): fix cache format or find programatic way to detect cache index
        # GPT2 and other models has a slightly different cache structure, with a different batch axis
        model_name = str(self.decoder) if "EncoderDecoder" in str(self) else str(self)  # type: ignore [attr-defined]
        cache_batch_axis = (
            1
            if any(model_prefix in model_name for model_prefix in ("TFGPT2", "TFCTRL"))
            else 0
        )
        # some models, like XLNet, need more than the last token in the presence of past_key_values
        needs_full_input = "use_mems" in set(
            inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
        )

        # 2. init `attentions`, `hidden_states`, and `scores` tuples
        scores = [] if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            [] if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            [] if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            [] if (return_dict_in_generate and output_hidden_states) else None
        )

        # 3. init tensors to use for "xla-compileable" generate function
        batch_size, cur_len = shape_list(input_ids)

        # initialize `generated` (pre-populated with `pad_token_id`), `finished_sequences`
        input_ids_padding = tf.ones(
            (batch_size, max_length - cur_len), dtype=tf.int32
        ) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        finished_sequences = tf.zeros((batch_size,), dtype=tf.bool)

        # 4. define "xla-compile-able" stop-condition and auto-regressive function
        def sample_cond_fn(generated, finished_sequences, cur_len, model_kwargs):
            return ~tf.reduce_all(finished_sequences)

        def sample_body_fn(generated, finished_sequences, cur_len, model_kwargs):
            if model_kwargs.get("past_key_values") is None or needs_full_input:
                input_ids = generated[:, :cur_len]
            else:
                input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, use_cache=use_cache, **model_kwargs
            )
            # forward pass to get next token logits
            assert isinstance(self, Callable)
            model_outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = model_outputs.logits[:, -1]

            # pre-process distribution
            next_tokens_scores = logits_processor(generated, next_token_logits, cur_len)
            next_tokens_scores = logits_warper(generated, next_tokens_scores, cur_len)

            # Store scores, attentions and hidden_states when required
            if not use_xla and return_dict_in_generate:
                if output_scores:
                    scores.append(next_tokens_scores)  # type: ignore [arg-type]
                if output_attentions and self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.decoder_attentions)  # type: ignore [arg-type]
                elif output_attentions and not self.config.is_encoder_decoder:
                    decoder_attentions.append(model_outputs.attentions)  # type: ignore [arg-type]
                    if self.config.is_encoder_decoder:
                        cross_attentions.append(model_outputs.cross_attentions)  # type: ignore [arg-type]

                if output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.decoder_hidden_states)  # type: ignore [arg-type]
                elif output_hidden_states and self.config.is_encoder_decoder:
                    decoder_hidden_states.append(model_outputs.hidden_states)  # type: ignore [arg-type]

            # sample
            if seed is not None:
                sample_seed = seed
            else:
                sample_seed = tf.experimental.numpy.random.randint(
                    tf.int32.min,
                    tf.int32.max,
                    (2,),
                    dtype=tf.int32,  # type: ignore [arg-type]
                )

            sampler = PSETFMixin.make_sampler(1, sample_seed)
            next_tokens = pse_sample_fn(next_tokens_scores, sampler)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)  # type: ignore [arg-type]
                next_tokens = next_tokens * unfinished_seq + pad_token_id * (
                    1 - unfinished_seq
                )
                next_token_is_eos = tf.math.reduce_any(
                    tf.equal(
                        tf.broadcast_to(next_tokens, (len(eos_token_id), batch_size)),  # type: ignore [arg-type]
                        tf.expand_dims(eos_token_id, -1),
                    ),
                    axis=0,
                )
                finished_sequences = finished_sequences | next_token_is_eos

            # update `generated` and `cur_len`
            update_indices = tf.stack(
                [tf.range(batch_size), tf.broadcast_to(cur_len, [batch_size])], axis=-1
            )
            generated = tf.tensor_scatter_nd_update(
                tensor=generated, indices=update_indices, updates=next_tokens
            )
            cur_len += 1

            # update model_kwargs
            if use_xla:
                model_kwargs = self._update_model_kwargs_for_xla_generation(
                    model_outputs=model_outputs,
                    model_kwargs=model_kwargs,
                    cur_len=cur_len,
                    max_length=max_length,
                    batch_size=batch_size,
                    is_encoder_decoder=self.config.is_encoder_decoder,  # type: ignore [attr-defined]
                    batch_axis=cache_batch_axis,
                )
            else:
                model_kwargs = self._update_model_kwargs_for_generation(
                    model_outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,  # type: ignore [attr-defined]
                )
                # if we don't cache past_key_values key values we need the whole input
                if model_kwargs.get("past_key_values", None) is None:
                    # let's throw out `past_key_values` since we don't want `None` tensors
                    model_kwargs.pop("past_key_values", None)

            return generated, finished_sequences, cur_len, model_kwargs

        # 5. run generation
        # 1st generation step has to be run before to initialize `past_key_values`
        generated, finished_sequences, cur_len, model_kwargs = sample_body_fn(
            generated, finished_sequences, cur_len, model_kwargs
        )

        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        maximum_iterations = max_length - cur_len
        generated, _, cur_len, _ = tf.while_loop(
            sample_cond_fn and not self.engine.has_reached_accept_state,
            sample_body_fn,
            (generated, finished_sequences, cur_len, model_kwargs),
            maximum_iterations=maximum_iterations,
        )  # type: ignore [arg-type]

        # 6. prepare outputs
        if not use_xla:
            # cut for backward compatibility
            generated = generated[:, :cur_len]

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:  # type: ignore [attr-defined]
                # if model is an encoder-decoder, retrieve encoder attention weights
                # and hidden states
                encoder_attentions = (
                    model_kwargs["encoder_outputs"].get("attentions")
                    if output_attentions
                    else None
                )
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states")
                    if output_hidden_states
                    else None
                )

                scores = tuple(scores) if scores is not None else None
                decoder_attentions = (
                    tuple(decoder_attentions)
                    if decoder_attentions is not None
                    else None
                )
                cross_attentions = (
                    tuple(cross_attentions) if cross_attentions is not None else None
                )
                decoder_hidden_states = (
                    tuple(decoder_hidden_states)
                    if decoder_hidden_states is not None
                    else None
                )

                return TFSampleEncoderDecoderOutput(
                    sequences=generated,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return TFSampleDecoderOnlyOutput(
                    sequences=generated,
                    scores=scores,  # type: ignore [arg-type]
                    attentions=decoder_attentions,  # type: ignore [arg-type]
                    hidden_states=decoder_hidden_states,  # type: ignore [arg-type]
                )
        else:
            return generated
