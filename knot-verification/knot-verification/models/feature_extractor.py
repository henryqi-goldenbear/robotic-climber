"""
Frozen DINOv2 ViT-Small feature extractor.

No gradient updates ever touch this model -- with only ~150 images, we let
DINOv2's self-supervised pretraining (which already encodes strand depth,
shadow, and continuous geometry) do all the representation learning, and
only train the tiny classifier head on top of its output.
"""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import config


class DinoV2FeatureExtractor:
    def __init__(self, model_name: str = config.DINO_MODEL_NAME, device: str = config.DEVICE):
        self.device = device
        # First call downloads pretrained weights from torch.hub's cache; no
        # repo clone needed.
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        self.transform = T.Compose([
            T.Resize((config.DINO_IMG_SIZE, config.DINO_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """Single crop -> (384,) class-token embedding."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.model(x)
        return feat.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, images) -> np.ndarray:
        """List of crops -> (N, 384) embeddings, batched for speed."""
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        feats = self.model(batch)
        return feats.cpu().numpy()
