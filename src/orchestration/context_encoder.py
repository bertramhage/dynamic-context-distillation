"""Frozen Chronos-2 context encoder for extracting hidden states from long history windows."""

import torch
from chronos import Chronos2Pipeline


class ChronosContextEncoder:
    """Wraps a frozen Chronos-2 model to extract encoder hidden states.

    Used by the orchestration layer to encode long history windows into
    feature representations that the hypernetwork consumes.
    """

    def __init__(self, pipeline: Chronos2Pipeline):
        self.model = pipeline.model
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Encode raw time-series values into hidden states via the Chronos-2 encoder.

        Args:
            context_tensor: [batch, seq_len] raw time-series values (one row per sensor).

        Returns:
            hidden_states: [batch, num_patches, d_model=768] encoder last hidden state.
        """
        context_tensor = context_tensor.to(self.device)
        batch_size = context_tensor.shape[0]

        encoder_outputs, _loc_scale, _reg_token, _num_ctx_patches = self.model.encode(
            context=context_tensor,
            context_mask=None,
            group_ids=torch.arange(batch_size, device=self.device),
            num_output_patches=1,
            output_attentions=False,
        )

        return encoder_outputs.last_hidden_state

    @torch.no_grad()
    def encode_batched(
        self, context_tensor: torch.Tensor, batch_size: int = 32
    ) -> torch.Tensor:
        """Encode in batches to limit GPU memory usage.

        Args:
            context_tensor: [n_sensors, seq_len] full tensor of all sensors.
            batch_size: max sensors per forward pass.

        Returns:
            hidden_states: [n_sensors, num_patches, d_model] concatenated results.
        """
        n_sensors = context_tensor.shape[0]
        hidden_chunks = []

        for start in range(0, n_sensors, batch_size):
            end = min(start + batch_size, n_sensors)
            chunk = context_tensor[start:end]
            hidden_chunks.append(self.encode(chunk))

        return torch.cat(hidden_chunks, dim=0)
