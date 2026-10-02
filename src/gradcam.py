"""Minimal Grad-CAM (no extra library)."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.act = model, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o))

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) tensor. Returns (cam HxW in [0,1], softmax probs, class index)."""
        self.model.zero_grad()
        logits = self.model(x)
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        grads = torch.autograd.grad(logits[0, idx], self.act)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)               # channel importance
        cam = torch.relu((weights * self.act).sum(1, keepdim=True))  # weighted activation map
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), torch.softmax(logits.detach(), 1)[0].cpu().numpy(), idx


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)
