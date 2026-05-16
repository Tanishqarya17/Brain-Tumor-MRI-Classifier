"""
Inference and Grad-CAM utilities for brain tumor MRI classification.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import (CLASS_NAMES, IDX_TO_CLASS, IMG_SIZE,
                     get_eval_transform)

eval_transform = get_eval_transform()


def predict(model, pil_image, device="cpu"):
    """
    Run the model on a single PIL image.

    Returns a dict with predicted class, confidence, and all probabilities.
    """
    pil_image = pil_image.convert("RGB")
    input_tensor = eval_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    return {
        "predicted_class": IDX_TO_CLASS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {IDX_TO_CLASS[i]: float(probs[i])
                          for i in range(len(CLASS_NAMES))},
    }


def generate_gradcam(model, pil_image, target_layer, device="cpu",
                     target_class=None):
    """
    Generate a Grad-CAM overlay for a single image.

    Args:
        model: the trained model
        pil_image: input PIL image
        target_layer: the conv layer to compute Grad-CAM on
                       (e.g. model.blocks[-1])
        device: 'cpu' or 'cuda'
        target_class: class index to explain; if None, uses prediction

    Returns:
        overlay image as a numpy array (H, W, 3), uint8
    """
    from torchvision import transforms

    pil_image = pil_image.convert("RGB")
    input_tensor = eval_transform(pil_image).unsqueeze(0).to(device)

    # Build a display image (no normalization)
    display_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    rgb = display_tf(pil_image).permute(1, 2, 0).numpy()

    if target_class is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target_class = int(logits.argmax())

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(target_class)])[0]
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return overlay
