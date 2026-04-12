import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from chronos.chronos_bolt import ChronosBoltModelForForecasting


class ChronosContextEncoder:
    """Runs a frozen Chronos encoder and captures hidden states.

    Supports both Chronos-2 (custom encoder blocks) and Chronos-Bolt
    (T5-based encoder with a clean encode() method).
    """

    def __init__(self, pipeline: BaseChronosPipeline):
        self.model = pipeline.model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self._is_bolt = isinstance(self.model, ChronosBoltModelForForecasting)

    @torch.no_grad()
    def encode_last_hidden(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Get last-layer hidden states. Returns [B, num_patches, d_model].

        For Chronos-Bolt models, uses the built-in encode() method.
        For Chronos-2 models, runs the full encoder and returns the final output.
        """
        context_tensor = context_tensor.to(self.device)

        if self._is_bolt:
            # ChronosBolt has a clean encode() returning (hidden_states, loc_scale, input_embeds, attention_mask)
            hidden_states, _loc_scale, _input_embeds, _attention_mask = self.model.encode(
                context=context_tensor
            )
            # hidden_states shape: [B, num_context_patches + (1 if REG), d_model]
            # Exclude the REG token (appended at the end) if present
            num_context_patches = hidden_states.shape[1]
            if self.model.chronos_config.use_reg_token:
                num_context_patches -= 1
            return hidden_states[:, :num_context_patches, :]

        # Chronos-2: run full encoder, return final hidden states (context patches only)
        model = self.model
        patched_context, attention_mask, loc_scale = model._prepare_patched_context(
            context=context_tensor, context_mask=None
        )
        num_context_patches = attention_mask.shape[-1]
        input_embeds = model.input_patch_embedding(patched_context)

        batch_size = context_tensor.shape[0]
        if model.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), model.config.reg_token_id, device=self.device
            )
            reg_embeds = model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(model.dtype),
                 torch.ones_like(reg_input_ids).to(model.dtype)],
                dim=-1,
            )

        patched_future, _ = model._prepare_patched_future(
            future_covariates=None, future_covariates_mask=None,
            loc_scale=loc_scale, num_output_patches=1, batch_size=batch_size,
        )
        future_embeds = model.input_patch_embedding(patched_future)
        future_mask = torch.ones(batch_size, 1, dtype=model.dtype, device=self.device)

        all_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        all_mask = torch.cat([attention_mask.to(model.dtype), future_mask], dim=-1)

        encoder = model.encoder
        group_ids = torch.arange(batch_size, device=self.device)
        extended_attn_mask = encoder._expand_and_invert_time_attention_mask(
            all_mask, all_embeds.dtype
        )
        group_time_mask = encoder._construct_and_invert_group_time_mask(
            group_ids, all_mask, all_embeds.dtype
        )
        seq_length = all_embeds.shape[1]
        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        hidden_states = encoder.dropout(all_embeds)
        for block in encoder.block:
            block_output = block(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attn_mask,
                group_time_mask=group_time_mask,
                output_attentions=False,
            )
            hidden_states = block_output[0]

        return hidden_states[:, :num_context_patches, :]

    @torch.no_grad()
    def encode_last_hidden_batched(
        self, context_tensor: torch.Tensor, batch_size: int = 32
    ) -> torch.Tensor:
        """Encode in sensor-batches and return last hidden states.

        Args:
            context_tensor: [n_sensors, seq_len] full sensor batch.
            batch_size: max sensors per forward pass.

        Returns:
            [n_sensors, num_context_patches, d_model]
        """
        n_sensors = context_tensor.shape[0]
        chunks: list[torch.Tensor] = []

        for start in range(0, n_sensors, batch_size):
            end = min(start + batch_size, n_sensors)
            chunks.append(self.encode_last_hidden(context_tensor[start:end]))

        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_intermediates(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Collect intermediate hidden states for a batch of time series.

        Args:
            context_tensor: [batch, seq_len] raw time-series values.

        Returns:
            all_z: [batch, num_layers, num_context_patches, d_model=768]
                   where all_z[:, l, :, :] = Z_l = input to encoder block l.
        """
        model = self.model
        encoder = model.encoder
        batch_size = context_tensor.shape[0]

        context_tensor = context_tensor.to(self.device)

        # --- Step 1: patch + scale context ---
        patched_context, attention_mask, loc_scale = model._prepare_patched_context(
            context=context_tensor, context_mask=None
        )
        num_context_patches = attention_mask.shape[-1]

        # --- Step 2: patch embeddings ---
        input_embeds: torch.Tensor = model.input_patch_embedding(patched_context)
        # shape: [batch, num_context_patches, d_model]

        # --- Step 3: [REG] token (if the model uses one) ---
        if model.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), model.config.reg_token_id, device=self.device
            )
            reg_embeds = model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(model.dtype),
                    torch.ones_like(reg_input_ids).to(model.dtype),
                ],
                dim=-1,
            )

        # --- Step 4: dummy future patch (keeps encoder inputs consistent with inference) ---
        patched_future, _ = model._prepare_patched_future(
            future_covariates=None,
            future_covariates_mask=None,
            loc_scale=loc_scale,
            num_output_patches=1,
            batch_size=batch_size,
        )
        future_embeds: torch.Tensor = model.input_patch_embedding(patched_future)
        future_mask = torch.ones(batch_size, 1, dtype=model.dtype, device=self.device)

        # Concatenate: [context (+REG) | future]
        all_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        all_mask = torch.cat([attention_mask.to(model.dtype), future_mask], dim=-1)

        # --- Step 5: build encoder attention masks ---
        group_ids = torch.arange(batch_size, device=self.device)
        extended_attn_mask = encoder._expand_and_invert_time_attention_mask(
            all_mask, all_embeds.dtype
        )
        group_time_mask = encoder._construct_and_invert_group_time_mask(
            group_ids, all_mask, all_embeds.dtype
        )
        seq_length = all_embeds.shape[1]
        position_ids = torch.arange(
            0, seq_length, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # --- Step 6: manual block loop, capture Z_l before each block ---
        hidden_states = encoder.dropout(all_embeds)

        all_z: list[torch.Tensor] = []
        for block in encoder.block:
            # Z_l = hidden state entering block l — context patches only.
            all_z.append(hidden_states[:, :num_context_patches, :].clone())

            block_output = block(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attn_mask,
                group_time_mask=group_time_mask,
                output_attentions=False,
            )
            hidden_states = block_output[0]

        # Stack along layer dimension → [batch, num_layers, num_context_patches, d_model]
        return torch.stack(all_z, dim=1)

    @torch.no_grad()
    def encode_intermediates_batched(
        self, context_tensor: torch.Tensor, batch_size: int = 32
    ) -> torch.Tensor:
        """Encode in sensor-batches to limit GPU memory usage.

        Args:
            context_tensor: [n_sensors, seq_len] full sensor batch.
            batch_size: max sensors per forward pass.

        Returns:
            [n_sensors, num_layers, num_context_patches, d_model]
        """
        n_sensors = context_tensor.shape[0]
        chunks: list[torch.Tensor] = []

        for start in range(0, n_sensors, batch_size):
            end = min(start + batch_size, n_sensors)
            chunks.append(self.encode_intermediates(context_tensor[start:end]))

        return torch.cat(chunks, dim=0)
