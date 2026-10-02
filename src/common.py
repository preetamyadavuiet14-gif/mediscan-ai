"""Shared helpers: transforms, model builder, Grad-CAM target layers, checkpoint loading."""
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ARCHS = ["mobilenet_v3_small", "resnet18", "resnet50", "densenet121", "efficientnet_b0"]


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalisation (improves lung-structure contrast)."""

    def __init__(self, clip=2.0, grid=8):
        self.clip, self.grid = clip, grid

    def __call__(self, img: Image.Image) -> Image.Image:
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.grid, self.grid))
        gray = clahe.apply(np.array(img.convert("L")))
        return Image.fromarray(gray).convert("RGB")


def train_tf():
    # No horizontal flip on purpose: heart position (left side) is anatomically meaningful.
    return T.Compose([
        CLAHE(),
        T.Resize((256, 256)),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_tf():
    return T.Compose([
        CLAHE(),
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def display_tf():
    """Same geometry as eval_tf but without normalisation, used for heatmap overlays."""
    return T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE)])


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    assert arch in ARCHS, f"arch must be one of {ARCHS}"
    model = models.get_model(arch, weights="DEFAULT" if pretrained else None)
    if arch.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif arch.startswith("densenet"):
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier.in_features, num_classes))
    elif arch.startswith("mobilenet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:  # efficientnet
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # In-place ops break Grad-CAM gradients, so switch them off.
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model


def get_head(model: nn.Module) -> nn.Module:
    return model.fc if hasattr(model, "fc") else model.classifier


def target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last convolutional block: where Grad-CAM looks."""
    if arch.startswith("resnet"):
        return model.layer4[-1]
    return model.features if arch.startswith("densenet") else model.features[-1]


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)  # Grad-CAM needs gradients through the network
    return model, ckpt
