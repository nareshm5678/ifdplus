import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    """Attention mechanism to focus on suspicious regions"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate attention map
        attention = self.sigmoid(self.conv(x))
        # Apply attention
        return x * attention.expand_as(x), attention

class ForgeryDetectionNet(nn.Module):
    """Image Forgery Detection Network with attention mechanism"""
    def __init__(self, num_classes=2, pretrained=True, with_segmentation=True):
        super(ForgeryDetectionNet, self).__init__()
        
        # Use a pre-trained ResNet as the backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avg pool and FC
        
        # Feature pyramid for multi-scale feature extraction
        self.pyramid_layers = nn.ModuleList([
            nn.Conv2d(2048, 256, kernel_size=1),  # P5
            nn.Conv2d(1024, 256, kernel_size=1),  # P4
            nn.Conv2d(512, 256, kernel_size=1),   # P3
        ])
        
        # Attention blocks for each pyramid level
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(256),
            AttentionBlock(256),
            AttentionBlock(256)
        ])
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Segmentation head (optional)
        self.with_segmentation = with_segmentation
        if with_segmentation:
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(256 * 3, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1),
                nn.Sigmoid()
            )
            
            # Upsampling layers for segmentation
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Get original image size for upsampling
        orig_size = x.size()[2:]
        
        # Extract features from backbone
        features = []
        x = self.backbone[0](x)  # Conv1
        x = self.backbone[1](x)  # BN1
        x = self.backbone[2](x)  # ReLU
        x = self.backbone[3](x)  # MaxPool
        
        x = self.backbone[4](x)  # Layer1
        features.append(x)  # Save for FPN
        
        x = self.backbone[5](x)  # Layer2
        features.append(x)  # Save for FPN
        
        x = self.backbone[6](x)  # Layer3
        features.append(x)  # Save for FPN
        
        x = self.backbone[7](x)  # Layer4
        features.append(x)  # Save for FPN
        
        # Apply feature pyramid
        pyramid_features = []
        attention_maps = []
        
        # Process P5 (from Layer4)
        p5 = self.pyramid_layers[0](features[3])
        p5, att5 = self.attention_blocks[0](p5)
        pyramid_features.append(p5)
        attention_maps.append(att5)
        
        # Process P4 (from Layer3)
        p4 = self.pyramid_layers[1](features[2])
        p4, att4 = self.attention_blocks[1](p4)
        pyramid_features.append(p4)
        attention_maps.append(att4)
        
        # Process P3 (from Layer2)
        p3 = self.pyramid_layers[2](features[1])
        p3, att3 = self.attention_blocks[2](p3)
        pyramid_features.append(p3)
        attention_maps.append(att3)
        
        # Global average pooling on each pyramid level
        pooled_features = [self.gap(feat) for feat in pyramid_features]
        pooled_features = [feat.view(feat.size(0), -1) for feat in pooled_features]
        
        # Concatenate pooled features
        concat_features = torch.cat(pooled_features, dim=1)
        
        # Classification
        classification = self.classifier(concat_features)
        
        if self.with_segmentation:
            # Upsample all pyramid features to the same size
            p3_size = pyramid_features[2].size()[2:]
            p5_up = F.interpolate(pyramid_features[0], size=p3_size, mode='bilinear', align_corners=True)
            p4_up = F.interpolate(pyramid_features[1], size=p3_size, mode='bilinear', align_corners=True)
            
            # Concatenate for segmentation
            seg_features = torch.cat([p5_up, p4_up, pyramid_features[2]], dim=1)
            
            # Apply segmentation head
            segmentation = self.segmentation_head(seg_features)
            
            # Upsample to original image size
            segmentation = F.interpolate(segmentation, size=orig_size, mode='bilinear', align_corners=True)
            
            return classification, segmentation, attention_maps
        else:
            return classification, attention_maps

class HybridForgeryDetector(nn.Module):
    """Hybrid model combining deep learning with traditional forensic features"""
    def __init__(self, num_classes=2, pretrained=True, with_segmentation=True):
        super(HybridForgeryDetector, self).__init__()
        
        # Deep learning branch
        self.dl_branch = ForgeryDetectionNet(num_classes=num_classes, 
                                           pretrained=pretrained,
                                           with_segmentation=with_segmentation)
        
        # Traditional forensic features branch
        # This would typically be implemented as a separate module or function
        # that extracts hand-crafted features like ELA, noise analysis, etc.
        self.forensic_features_dim = 64  # Dimension of traditional forensic features
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 + self.forensic_features_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.with_segmentation = with_segmentation
        
    def forward(self, x, forensic_features):
        # Process through deep learning branch
        if self.with_segmentation:
            dl_classification, segmentation, attention_maps = self.dl_branch(x)
        else:
            dl_classification, attention_maps = self.dl_branch(x)
        
        # Extract intermediate features before final classification
        dl_features = self.dl_branch.classifier[0].weight.mm(dl_classification.t()).t()
        
        # Concatenate with traditional forensic features
        combined_features = torch.cat([dl_features, forensic_features], dim=1)
        
        # Final classification through fusion layer
        final_classification = self.fusion(combined_features)
        
        if self.with_segmentation:
            return final_classification, segmentation, attention_maps
        else:
            return final_classification, attention_maps

# Loss functions
class ForgeryDetectionLoss(nn.Module):
    """Combined loss for classification and segmentation"""
    def __init__(self, seg_weight=1.0, att_weight=0.1):
        super(ForgeryDetectionLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.BCELoss()
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        
    def dice_loss(self, pred, target, smooth=1):
        """Compute Dice loss for segmentation quality."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice

    def forward(self, pred_class, true_class, pred_seg=None, true_seg=None, attention_maps=None):
        # Classification loss
        cls_loss = self.classification_loss(pred_class, true_class)
        
        total_loss = cls_loss
        
        # Add segmentation loss if applicable
        if pred_seg is not None and true_seg is not None:
            # Combine BCE loss with Dice loss for more precise segmentation
            bce_loss = self.segmentation_loss(pred_seg, true_seg)
            dice = self.dice_loss(pred_seg, true_seg)
            seg_loss = bce_loss + dice
            total_loss += self.seg_weight * seg_loss
        
        # Add attention regularization if applicable
        if attention_maps is not None:
            # Encourage sparsity in attention maps
            att_reg = sum(torch.mean(torch.abs(att)) for att in attention_maps)
            total_loss += self.att_weight * att_reg
        
        return total_loss