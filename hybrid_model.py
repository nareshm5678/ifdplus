import torch
import torch.nn as nn
import torch.nn.functional as F
from model_architecture import ForgeryDetectionNet
from forensic_features import ForensicFeatureExtractor

class HybridForgeryDetector(nn.Module):
    """Hybrid model combining deep learning with traditional forensic features"""
    def __init__(self, num_classes=2, pretrained=True, with_segmentation=True, forensic_feature_dim=64):
        super(HybridForgeryDetector, self).__init__()
        
        # Deep learning branch
        self.dl_branch = ForgeryDetectionNet(num_classes=num_classes, 
                                           pretrained=pretrained,
                                           with_segmentation=with_segmentation)
        
        # Traditional forensic features branch
        self.forensic_features_dim = forensic_feature_dim
        
        # Feature extraction layers for forensic features
        self.forensic_feature_extractor = nn.Sequential(
            nn.Linear(4 * 256 * 256, 1024),  # Assuming 4 feature maps of size 256x256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.forensic_features_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3 + self.forensic_features_dim, 512),  # 256*3 from the DL branch
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.with_segmentation = with_segmentation
        
    def forward(self, x, forensic_features):
        # Process through deep learning branch
        if self.with_segmentation:
            dl_outputs, segmentation, attention_maps = self.dl_branch(x)
        else:
            dl_outputs, attention_maps = self.dl_branch(x)
        
        # Extract intermediate features before final classification
        # Get the features from the global average pooling layer
        dl_features = []
        for i, layer in enumerate(self.dl_branch.backbone):
            if i <= 7:  # Up to Layer4
                x = layer(x)
                if i in [4, 5, 6, 7]:  # Layer1, Layer2, Layer3, Layer4
                    dl_features.append(x)
        
        # Apply pyramid and attention
        pyramid_features = []
        for i in range(3):  # P5, P4, P3
            p = self.dl_branch.pyramid_layers[i](dl_features[3-i])
            p, _ = self.dl_branch.attention_blocks[i](p)
            pyramid_features.append(p)
        
        # Global average pooling
        pooled_features = [self.dl_branch.gap(feat) for feat in pyramid_features]
        pooled_features = [feat.view(feat.size(0), -1) for feat in pooled_features]
        
        # Concatenate pooled features
        dl_concat_features = torch.cat(pooled_features, dim=1)
        
        # Process forensic features
        processed_forensic = self.forensic_feature_extractor(forensic_features.view(forensic_features.size(0), -1))
        
        # Concatenate deep learning and forensic features
        combined_features = torch.cat([dl_concat_features, processed_forensic], dim=1)
        
        # Final classification through fusion layer
        final_classification = self.fusion(combined_features)
        
        if self.with_segmentation:
            return final_classification, segmentation, attention_maps
        else:
            return final_classification, attention_maps

class HybridForgeryDetectionLoss(nn.Module):
    """Combined loss for hybrid forgery detection"""
    def __init__(self, seg_weight=1.0, att_weight=0.1, forensic_weight=0.5):
        super(HybridForgeryDetectionLoss, self).__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.BCELoss()
        self.forensic_loss = nn.MSELoss()  # For comparing forensic features
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.forensic_weight = forensic_weight
        
    def forward(self, pred_class, true_class, pred_seg=None, true_seg=None, 
                attention_maps=None, pred_forensic=None, true_forensic=None):
        # Classification loss
        cls_loss = self.classification_loss(pred_class, true_class)
        
        # Initialize total loss with classification loss
        total_loss = cls_loss
        
        # Add segmentation loss if available
        if pred_seg is not None and true_seg is not None:
            seg_loss = self.segmentation_loss(pred_seg, true_seg)
            total_loss += self.seg_weight * seg_loss
        
        # Add attention regularization if available
        if attention_maps is not None:
            # Encourage sparsity in attention maps
            att_loss = 0
            for att in attention_maps:
                att_loss += torch.mean(att)  # L1 regularization
            total_loss += self.att_weight * att_loss
        
        # Add forensic feature loss if available
        if pred_forensic is not None and true_forensic is not None:
            forensic_loss = self.forensic_loss(pred_forensic, true_forensic)
            total_loss += self.forensic_weight * forensic_loss
        
        return total_loss