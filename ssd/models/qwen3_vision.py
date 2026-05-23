"""Vision encoder wrapper for Qwen3-VL / Qwen2.5-VL models.

Qwen3-VL reuses the Qwen2-VL vision architecture (ViT with 3D RoPE).
This module wraps the HuggingFace vision encoder so it can be used
inside the SSD speculative decoding pipeline.
"""

import torch
from torch import nn


class Qwen3VisionEncoder(nn.Module):
    """Wraps the Qwen2-VL / Qwen3-VL vision transformer.

    Loads the vision tower from a HuggingFace VLM checkpoint and provides
    a simple forward interface: pixel_values + image_grid_thw -> image_embeds.
    """

    def __init__(self, hf_config, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dtype = dtype

        # The vision config lives under hf_config.vision_config
        vision_config = getattr(hf_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(
                "hf_config does not contain a vision_config. "
                "Make sure you are using a VLM checkpoint (e.g. Qwen2.5-VL or Qwen3-VL)."
            )

        # Import the HF vision model class
        try:
            from transformers import Qwen2VLForConditionalGeneration
            # Extract just the vision model architecture from a dummy or the config
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
            self.visual = Qwen2VisionTransformerPretrainedModel._from_config(vision_config)
        except ImportError:
            # Fallback: try Qwen2_5 naming
            try:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionModel
                self.visual = Qwen2_5_VLVisionModel(vision_config)
            except ImportError:
                raise ImportError(
                    "Could not import Qwen2-VL or Qwen2.5-VL vision model from transformers. "
                    "Please upgrade transformers: pip install transformers>=4.45.0"
                )

        # The visual projection maps vision hidden size -> LM hidden size
        vision_hidden_size = getattr(vision_config, "hidden_size", None)
        lm_hidden_size = getattr(hf_config, "hidden_size", None)
        if vision_hidden_size and lm_hidden_size and vision_hidden_size != lm_hidden_size:
            # Check if the VLM config already has a projection
            # Qwen2-VL uses a merger/projection inside the visual model
            pass  # Qwen2-VL handles projection internally in the visual model

        self.vision_hidden_size = vision_hidden_size
        self.lm_hidden_size = lm_hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Process images through the vision encoder.

        Args:
            pixel_values: Preprocessed image tensor from the Qwen2-VL processor.
                Shape depends on the processor but typically [num_patches, channels, H, W]
                or flattened.
            image_grid_thw: Tensor of shape [num_images, 3] containing
                (temporal, height_patches, width_patches) for each image.

        Returns:
            image_embeds: Tensor of shape [total_image_tokens, lm_hidden_size]
                Ready to be inserted into the text embedding sequence.
        """
        pixel_values = pixel_values.to(dtype=self.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds
