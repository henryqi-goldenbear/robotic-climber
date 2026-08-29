"""
Frozen DINOv2 ViT-Small feature extractor.

No gradient updates ever touch this model -- with only ~150 images, we let
DINOv2's self-supervised pretraining (which already encodes strand depth,
shadow, and continuous geometry) do all the representation learning, and
only train the tiny classifier head on top of its output.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import config


class DinoV2FeatureExtractor:
    def __init__(self, model_name: str = config.DINO_MODEL_NAME, device: str = config.DEVICE):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        self.processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """Single crop -> (384,) class-token embedding."""
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        feat = getattr(outputs, "pooler_output", None)
        if feat is None:
            feat = outputs.last_hidden_state[:, 0]
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, images) -> np.ndarray:
        """List of crops -> (N, 384) embeddings, batched for speed."""
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        feat = getattr(outputs, "pooler_output", None)
        if feat is None:
            feat = outputs.last_hidden_state[:, 0]
        return feat.cpu().numpy()
