
"""
Brain Tumor MRI Classification — Gradio Demo
Stage 2 EfficientNet-B3 trained with patient-level splitting.

For deployment on Hugging Face Spaces:
1. Upload this file as app.py
2. Upload stage2_best.pt to the Space root (or HF Hub)
3. requirements.txt:
     torch
     torchvision
     timm
     pytorch_grad_cam
     gradio>=4.0
     pillow
     numpy
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import gradio as gr

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ============================================================
# Configuration
# ============================================================
CLASS_NAMES   = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX  = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS  = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# Load model
# ============================================================
MODEL_PATH = "stage2_best.pt"  # adjust path if needed
print(f"Loading model from {MODEL_PATH}...")
model = timm.create_model("efficientnet_b3", pretrained=False,
                          num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Grad-CAM target layer (deepest conv block)
cam = GradCAM(model=model, target_layers=[model.blocks[-1]])

# ============================================================
# Transforms
# ============================================================
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

# ============================================================
# Prediction function
# ============================================================
def predict(input_image):
    if input_image is None:
        return (
            {name: 0.0 for name in CLASS_NAMES},
            np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
            "**Please upload an image or click an example below.**"
        )

    if isinstance(input_image, np.ndarray):
        if input_image.ndim == 2:
            input_image = np.stack([input_image] * 3, axis=-1)
        elif input_image.ndim == 3 and input_image.shape[2] == 4:
            input_image = input_image[:, :, :3]
        pil_image = Image.fromarray(input_image).convert("RGB")
    else:
        pil_image = input_image.convert("RGB")

    input_tensor = eval_transform(pil_image).unsqueeze(0).to(DEVICE)
    display_tensor = display_transform(pil_image)
    rgb_image = display_tensor.permute(1, 2, 0).numpy()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        predicted_class = int(probabilities.argmax())
        predicted_name = IDX_TO_CLASS[predicted_class]
        confidence = float(probabilities[predicted_class])

    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    probabilities_dict = {
        IDX_TO_CLASS[i]: float(probabilities[i])
        for i in range(NUM_CLASSES)
    }

    if confidence >= 0.90:
        conf_label = "High confidence"
        conf_emoji = "🟢"
    elif confidence >= 0.70:
        conf_label = "Moderate confidence"
        conf_emoji = "🟡"
    else:
        conf_label = "Low confidence — interpret with caution"
        conf_emoji = "🟠"

    sorted_idx = np.argsort(probabilities)[::-1]
    second_name = IDX_TO_CLASS[sorted_idx[1]]
    second_prob = probabilities[sorted_idx[1]]

    summary = f"""
### Prediction: **{predicted_name.upper()}**

**Confidence:** {confidence:.1%} {conf_emoji} {conf_label}

**Second most likely:** {second_name} ({second_prob:.1%})

The Grad-CAM heatmap on the right shows which regions of the image
most influenced this prediction. Red/yellow = high attention,
blue/green = low attention.
"""

    return probabilities_dict, overlay, summary


# ============================================================
# Disclaimer / description
# ============================================================
DISCLAIMER_MD = """
# Brain Tumor MRI Classification — Interactive Demo

> ⚠️ **MEDICAL DISCLAIMER:**
> This is a portfolio/research demonstration and **must NOT be used
> for any medical decision-making.** The model has not been validated
> in a clinical setting, has not been reviewed by radiologists, and is
> trained on a limited dataset from specific scanners.
> For medical concerns, consult a qualified healthcare professional.

This app classifies brain MRI scans into four categories:
**glioma**, **meningioma**, **pituitary tumor**, or **no tumor**.

It also generates a **Grad-CAM heatmap** showing where the model
focused its attention.

**How to use:** Click an example image below or upload your own brain MRI.

📊 **Model:** EfficientNet-B3, patient-level splitting (zero data leakage).
**Test accuracy:** 95.05% with TTA across 687 unseen patients.
**Test AUC:** 0.9965 macro-averaged.

[Full project details on GitHub →](https://github.com/Tanishqarya17)
"""

# ============================================================
# Example gallery
# ============================================================
# Examples must exist in the same folder as app.py
# Folder structure on HF Spaces:
#   ./app.py
#   ./stage2_best.pt
#   ./examples/
#       example_00_*.png
#       example_01_*.png
#       ...

EXAMPLES_DIR = "examples"
if os.path.exists(EXAMPLES_DIR):
    example_paths = sorted([
        os.path.join(EXAMPLES_DIR, f)
        for f in os.listdir(EXAMPLES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    gradio_examples = [[p] for p in example_paths]
else:
    gradio_examples = None
print(f"Examples found: {len(gradio_examples) if gradio_examples else 0}")

# ============================================================
# Build & launch app
# ============================================================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Brain MRI Image", height=350),
    outputs=[
        gr.Label(num_top_classes=NUM_CLASSES,
                 label="Prediction Confidence by Class"),
        gr.Image(type="numpy",
                 label="Grad-CAM Heatmap (red = model attention)",
                 height=350),
        gr.Markdown(label="Prediction Details"),
    ],
    title="Brain Tumor MRI Classification",
    description=DISCLAIMER_MD,
    article=(
        "### About this project\n\n"
        "This demo uses patient-level data splitting — a method that "
        "prevents the same patient's MRI slices from appearing in both "
        "training and test sets. Most public brain tumor classifiers "
        "don't do this, which inflates their accuracy by 5-15 percentage "
        "points.\n\n"
        "**Honest test accuracy: 95.05%** (TTA) on 687 patients the model "
        "has never seen.\n\n"
        "**Known limitation:** Lower accuracy on meningioma (80.4% recall) "
        "due to fewer unique training patients and visual similarity with "
        "glioma on single-modality MRI.\n\n"
        "Built with PyTorch + EfficientNet-B3 + Grad-CAM."
    ),
    examples=gradio_examples,
    cache_examples=False,
    css=".gradio-container h1 { text-align: center; }",
    
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"))

