# S2-CLIP image features

import torch
from s2wrapper import forward as multiscale_forward
from transformers import AutoProcessor, CLIPVisionModel

CLIP_MODEL = "openai/clip-vit-base-patch32"
S2_SCALES = [1, 2]


class S2CLIP:
    def __init__(self, device):
        self.device = device
        self.model = CLIPVisionModel.from_pretrained(CLIP_MODEL).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(CLIP_MODEL)

    @torch.no_grad()
    def __call__(self, image):
        """PIL image -> 1-D float tensor (on CPU)."""
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        features = multiscale_forward(
            lambda x: self.model(x).last_hidden_state,
            pixel_values.to(self.device),
            scales=S2_SCALES,
            num_prefix_token=1,  # the [CLS] token
        )
        return features.flatten().cpu()
