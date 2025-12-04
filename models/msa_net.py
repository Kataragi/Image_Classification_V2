"""
MSA-Net: Multimodal Style Aggregation Network
Multi-scale feature aggregation module for enhanced art style classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAttention(nn.Module):
    """Multi-scale attention module for style feature aggregation"""

    def __init__(self, in_channels, reduction=16):
        super(MultiScaleAttention, self).__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        # Spatial attention
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(batch, channels))
        max_out = self.fc(self.max_pool(x).view(batch, channels))
        channel_att = torch.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)
        x = x * channel_att

        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1))
        x = x * spatial_att

        return x


class StyleAggregationModule(nn.Module):
    """Aggregates multi-scale style features"""

    def __init__(self, feature_dim):
        super(StyleAggregationModule, self).__init__()

        self.attention = MultiScaleAttention(feature_dim)

        # Style-specific convolutions
        self.texture_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.color_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Apply attention
        x_att = self.attention(x)

        # Multi-branch processing
        texture_feat = self.texture_branch(x_att)
        color_feat = self.color_branch(x_att)

        # Fuse features
        fused = torch.cat([texture_feat, color_feat], dim=1)
        out = self.fusion(fused)

        return out


class MSANet(nn.Module):
    """
    MSA-Net wrapper for ConvNeXtV2
    Adds multi-scale style aggregation capabilities
    """

    def __init__(self, base_model, num_classes):
        super(MSANet, self).__init__()

        self.base_model = base_model

        # Get feature dimension from base model
        # ConvNeXtV2 has 1024 features before the classifier
        self.feature_dim = 1024

        # Replace the original classifier head
        # ConvNeXtV2 structure: features -> head (norm + flatten + fc)
        if hasattr(base_model, 'head'):
            if hasattr(base_model.head, 'fc'):
                base_model.head.fc = nn.Identity()
            else:
                base_model.head = nn.Identity()

        # Style aggregation module
        self.style_aggregation = StyleAggregationModule(self.feature_dim)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x):
        """Extract features for visualization"""
        # Get features from base model
        if hasattr(self.base_model, 'forward_features'):
            features = self.base_model.forward_features(x)
        else:
            features = self.base_model(x)

        # If features are 1D, we need to reshape for convolution
        # ConvNeXtV2 returns (B, C, H, W) so this should be fine
        if features.dim() == 2:
            # Reshape to (B, C, 1, 1) if needed
            features = features.unsqueeze(-1).unsqueeze(-1)

        # Apply style aggregation
        style_features = self.style_aggregation(features)

        # Global pooling
        pooled_features = self.global_pool(style_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        return pooled_features

    def forward(self, x):
        # Extract features
        features = self.extract_features(x)

        # Classify
        output = self.classifier(features)

        return output
