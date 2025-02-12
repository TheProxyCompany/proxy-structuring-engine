from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from transformers import FlaxGenerationMixin, FlaxLogitsProcessorList
from transformers.generation.flax_utils import FlaxSampleOutput, SampleState

from pse.engine.structuring_engine import StructuringEngine


class PSE_FlaxMixin(FlaxGenerationMixin):
    engine: StructuringEngine

    def make_sampler(self, prng_key: jnp.ndarray) -> Callable:
        return lambda x: jax.random.categorical(prng_key, x, axis=-1)

    def _sample(
        self,
        input_ids: None,
        max_length: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        prng_key: jnp.ndarray | None = None,
        logits_processor: FlaxLogitsProcessorList | None = None,
        logits_warper: FlaxLogitsProcessorList | None = None,
        trace: bool = True,
        params: dict[str, jnp.ndarray] | None = None,
        model_kwargs: dict[str, jnp.ndarray] | None = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape # type: ignore [arg-type]

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32) if eos_token_id is not None else None # type: ignore [arg-type]
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32) # type: ignore [arg-type]
        cur_len = jnp.array(cur_len) # type: ignore [arg-type]

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32) # type: ignore [arg-type]
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0)) # type: ignore [arg-type]

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        if not logits_processor or self.engine.process_logits not in logits_processor:
            # insert the engine at the beginning of the list
            if logits_processor is None:
                logits_processor = FlaxLogitsProcessorList()
            logits_processor.insert(0, self.engine.process_logits)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self # type: ignore [attr-defined]

        assert isinstance(model, Callable)
        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs) # type: ignore [arg-type]

        # initialize state
        state = SampleState(
            cur_len=cur_len, # type: ignore [arg-type]
            sequences=sequences, # type: ignore [arg-type]
            running_token=input_ids, # type: ignore [arg-type]
            is_sent_finished=is_sent_finished, # type: ignore [arg-type]
            prng_key=prng_key, # type: ignore [arg-type]
            model_kwargs=model_kwargs, # type: ignore [arg-type]
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]
            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_p, top_k, temperature
            if logits_warper:
                logits = logits_warper(logits, logits, state.cur_len)

            sampler = self.make_sampler(prng_key)
            next_token = self.engine.sample(logits, sampler)

            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs) # type: ignore [attr-defined]

            return SampleState(
                cur_len=state.cur_len + len(next_token), # type: ignore [arg-type]
                sequences=next_sequences, # type: ignore [arg-type]
                running_token=next_token, # type: ignore [arg-type]
                is_sent_finished=next_is_sent_finished, # type: ignore [arg-type]
                model_kwargs=next_model_kwargs, # type: ignore [arg-type]
                prng_key=prng_key_next, # type: ignore [arg-type]
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1: # type: ignore [arg-type]
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(
                lambda state: sample_search_cond_fn(state) and not self.engine.has_reached_accept_state,
                sample_search_body_fn,
                state,
            )
        else:
            state = lax.while_loop(
                lambda state: sample_search_cond_fn(state) and not self.engine.has_reached_accept_state,
                sample_search_body_fn,
                state,
            )

        return FlaxSampleOutput(sequences=state.sequences)
