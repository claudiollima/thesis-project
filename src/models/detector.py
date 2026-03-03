"""
DeepFake Image Detector

Based on ConvNeXt-V2 architecture as used in recent deepfake detection papers.
Reference: "OpenFake: An Open Dataset and Platform Toward Large-Scale Deepfake Detection"
"""

import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    """
    Binary classifier for detecting AI-generated images.
    
    Uses ConvNeXt-V2-Base backbone with a custom classification head.
    """

    def __init__(
        self,
        backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[-1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            logits: Raw predictions [B, 1]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability that image is fake."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


class LightweightDetector(nn.Module):
    """
    Smaller, faster detector for initial experiments.
    
    Uses EfficientNet-B0 backbone.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0.ra_in1k",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.feature_dim = self.backbone(dummy).shape[-1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def create_detector(model_type: str = "convnext", **kwargs) -> nn.Module:
    """Factory function to create detector models."""
    if model_type == "convnext":
        return DeepfakeDetector(**kwargs)
    elif model_type == "efficientnet":
        return LightweightDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
