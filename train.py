import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import cv2
import random
import time
from model_architecture import ForgeryDetectionNet, ForgeryDetectionLoss

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Dataset paths
DATASET_PATH = os.path.join(os.getcwd(), 'Casiadataset')
IMAGE_PATH = os.path.join(DATASET_PATH, 'IMAGE')
MASK_PATH = os.path.join(DATASET_PATH, 'MASK')

# Authentic and tampered image paths
AU_PATH = os.path.join(IMAGE_PATH, 'Au')
TP_PATH = os.path.join(IMAGE_PATH, 'Tp')

# Mask paths
AU_MASK_PATH = os.path.join(MASK_PATH, 'Au') if os.path.exists(os.path.join(MASK_PATH, 'Au')) else None
TP_MASK_PATH = os.path.join(MASK_PATH, 'Tp')

class CASIADataset(Dataset):
    """CASIA Dataset for Image Forgery Detection"""
    def __init__(self, au_path, tp_path, au_mask_path, tp_mask_path, transform=None, mask_transform=None, split='train', train_ratio=0.8):
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split
        
        # Get all authentic and tampered image paths
        self.au_images = [os.path.join(au_path, f) for f in os.listdir(au_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.tp_images = [os.path.join(tp_path, f) for f in os.listdir(tp_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Get corresponding mask paths
        self.au_masks = []
        if au_mask_path and os.path.exists(au_mask_path):
            self.au_masks = [os.path.join(au_mask_path, f) for f in os.listdir(au_mask_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.tp_masks = [os.path.join(tp_mask_path, f) for f in os.listdir(tp_mask_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create image-mask pairs for tampered images
        self.tp_pairs = []
        for tp_img in self.tp_images:
            img_name = os.path.basename(tp_img)
            img_base = os.path.splitext(img_name)[0]
            
            # Find corresponding mask
            mask_found = False
            for tp_mask in self.tp_masks:
                mask_name = os.path.basename(tp_mask)
                mask_base = os.path.splitext(mask_name)[0]
                
                if mask_base == img_base:
                    self.tp_pairs.append((tp_img, tp_mask))
                    mask_found = True
                    break
            
            # If no mask found, use a blank mask
            if not mask_found:
                self.tp_pairs.append((tp_img, None))
        
        # Split data for training and validation
        random.shuffle(self.au_images)
        random.shuffle(self.tp_pairs)
        
        au_split = int(len(self.au_images) * train_ratio)
        tp_split = int(len(self.tp_pairs) * train_ratio)
        
        if split == 'train':
            self.au_images = self.au_images[:au_split]
            self.tp_pairs = self.tp_pairs[:tp_split]
        else:  # validation
            self.au_images = self.au_images[au_split:]
            self.tp_pairs = self.tp_pairs[tp_split:]
        
        # Balance the dataset
        min_len = min(len(self.au_images), len(self.tp_pairs))
        self.au_images = self.au_images[:min_len]
        self.tp_pairs = self.tp_pairs[:min_len]
        
        print(f"{split} set: {len(self.au_images)} authentic images, {len(self.tp_pairs)} tampered images")
    
    def __len__(self):
        return len(self.au_images) + len(self.tp_pairs)
    
    def __getitem__(self, idx):
        # Authentic image (label 0)
        if idx < len(self.au_images):
            img_path = self.au_images[idx]
            label = 0
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Create blank mask for authentic images
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            mask = Image.fromarray(mask)
            
        # Tampered image (label 1)
        else:
            idx = idx - len(self.au_images)
            img_path, mask_path = self.tp_pairs[idx]
            label = 1
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Load mask if available, otherwise create blank mask
            if mask_path:
                mask = Image.open(mask_path).convert('L')
                # Ensure mask is binary (0 or 255)
                mask = np.array(mask)
                mask = (mask > 128).astype(np.uint8) * 255
                mask = Image.fromarray(mask)
            else:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                mask = Image.fromarray(mask)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask, label

# Data augmentation and preprocessing
def get_transforms(img_size=256):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    return train_transform, val_transform, mask_transform

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=2, device='cuda'):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_iou': [], 'val_dice': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, masks, labels in dataloader:
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, segmentations, attention_maps = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels, segmentations, masks, attention_maps)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
        
        print()
        
        # After each epoch, compute IoU and Dice on validation set
        metrics = evaluate_model(model, val_loader, device)
        val_iou = metrics['mean_iou']
        val_dice = metrics['mean_dice']
        print(f"Epoch {epoch+1} Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        
        # Deep copy the model if best validation accuracy
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict().copy()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

# Evaluation function
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_segmentations = []
    all_masks = []
    
    with torch.no_grad():
        for inputs, masks, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, segmentations, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Binarize and squeeze predicted segmentation masks
            seg_np = segmentations.cpu().numpy()
            seg_np = (seg_np > 0.5).astype(np.uint8).squeeze(1)
            all_segmentations.extend(seg_np)
            # Binarize and squeeze ground truth masks
            gt_masks = masks.cpu().numpy().astype(np.uint8).squeeze(1)
            all_masks.extend(gt_masks)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Calculate IoU for segmentation (for tampered images only)
    tampered_indices = [i for i, label in enumerate(all_labels) if label == 1]
    if tampered_indices:
        tampered_segmentations = [all_segmentations[i] for i in tampered_indices]
        tampered_masks = [all_masks[i] for i in tampered_indices]
        
        # Convert to binary masks
        tampered_segmentations = [(seg > 0.5).astype(np.uint8) for seg in tampered_segmentations]
        tampered_masks = [(mask > 0.5).astype(np.uint8) for mask in tampered_masks]
        
        # Calculate IoU
        ious = []
        for seg, mask in zip(tampered_segmentations, tampered_masks):
            intersection = np.logical_and(seg, mask).sum()
            union = np.logical_or(seg, mask).sum()
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        mean_iou = np.mean(ious)
        print(f"Mean IoU for tampered regions: {mean_iou:.4f}")
        
        # Compute Dice scores for tampered regions
        dice_scores = []
        for seg, mask in zip(tampered_segmentations, tampered_masks):
            inter = np.logical_and(seg, mask).sum()
            denom = seg.sum() + mask.sum()
            dice_scores.append((2 * inter) / denom if denom > 0 else 0)
        mean_dice = np.mean(dice_scores)
        print(f"Mean Dice for tampered regions: {mean_dice:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'mean_iou': mean_iou if tampered_indices else 0,
        'mean_dice': mean_dice if tampered_indices else 0
    }

# Visualization function
def visualize_results(model, test_loader, num_samples=5, device='cuda'):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4*num_samples))
    
    with torch.no_grad():
        for i, (inputs, masks, labels) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, segmentations, attention_maps = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Convert tensors to numpy arrays
            input_img = inputs[0].cpu().permute(1, 2, 0).numpy()
            input_img = (input_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            
            mask = masks[0].cpu().squeeze().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            pred_mask = segmentations[0].cpu().squeeze().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            
            # Get attention map (combine all attention maps) - Fix for tensor size mismatch
            # Use only the first attention map for visualization
            attention = attention_maps[0][0].cpu().squeeze().numpy()
            
            # Plot
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title(f"Input (True: {labels[0].item()}, Pred: {preds[0].item()})") 
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(attention, cmap='jet')
            axes[i, 3].set_title("Attention Map")
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png')
    plt.close()
    print("Results visualization saved as 'results_visualization.png'")

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'")

# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set image size and batch size
    img_size = 256
    batch_size = 8
    
    # Get transforms
    train_transform, val_transform, mask_transform = get_transforms(img_size)
    
    # Create datasets
    train_dataset = CASIADataset(
        au_path=AU_PATH,
        tp_path=TP_PATH,
        au_mask_path=AU_MASK_PATH,
        tp_mask_path=TP_MASK_PATH,
        transform=train_transform,
        mask_transform=mask_transform,
        split='train'
    )
    
    val_dataset = CASIADataset(
        au_path=AU_PATH,
        tp_path=TP_PATH,
        au_mask_path=AU_MASK_PATH,
        tp_mask_path=TP_MASK_PATH,
        transform=val_transform,
        mask_transform=mask_transform,
        split='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = ForgeryDetectionNet(num_classes=2, pretrained=True, with_segmentation=True)
    model = model.to(device)
    
    # Define loss function and optimizer (increased segmentation weight)
    criterion = ForgeryDetectionLoss(seg_weight=2.0, att_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,  # Reduced from 25 to 2 for faster testing
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), 'forgery_detection_model_50.pth')
    print("Model saved as 'forgery_detection_model_50.pth'")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    metrics = evaluate_model(model, val_loader, device)
    
    # Visualize results
    visualize_results(model, val_loader, num_samples=5, device=device)

if __name__ == "__main__":
    main()