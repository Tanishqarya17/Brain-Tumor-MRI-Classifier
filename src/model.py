"""
Model construction for brain tumor MRI classification.
EfficientNet-B3 backbone via the timm library.
"""

import torch
import timm

NUM_CLASSES = 4


def build_model(pretrained=False):
    """
    Build an EfficientNet-B3 model with a 4-class head.

    Args:
        pretrained: if True, load ImageNet pretrained weights.
                    Use False when loading your own trained checkpoint.

    Returns:
        a torch.nn.Module
    """
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
    return model


def load_trained_model(checkpoint_path, device="cpu"):
    """
    Build the model and load trained weights from a checkpoint.

    Args:
        checkpoint_path: path to the .pt state_dict file
        device: 'cpu' or 'cuda'

    Returns:
        model in eval mode on the requested device
    """
    model = build_model(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
