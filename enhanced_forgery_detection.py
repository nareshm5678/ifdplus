import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model_architecture import ForgeryDetectionNet, ForgeryDetectionLoss
from hybrid_model import HybridForgeryDetector, HybridForgeryDetectionLoss
from forensic_features import ForensicFeatureExtractor, visualize_ela
from adversarial_training import AdversarialTrainer, test_robustness
from explainable_ai import explain_prediction, explain_batch
from train import CASIADataset, get_transforms
from inference import process_image as standard_process_image
from hybrid_inference import process_image as hybrid_process_image

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

def train_enhanced_model(model_type='hybrid', use_adversarial=True, num_epochs=25, batch_size=16, device='cuda'):
    """Train an enhanced forgery detection model"""
    print(f"Training {model_type} model {'with' if use_adversarial else 'without'} adversarial training...")
    
    # Set image size
    img_size = 256
    
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
    if model_type == 'hybrid':
        model = HybridForgeryDetector(num_classes=2, pretrained=True, with_segmentation=True)
        criterion = HybridForgeryDetectionLoss(seg_weight=1.0, att_weight=0.1, forensic_weight=0.5)
    else:
        model = ForgeryDetectionNet(num_classes=2, pretrained=True, with_segmentation=True)
        criterion = ForgeryDetectionLoss(seg_weight=1.0, att_weight=0.1)
    
    model = model.to(device)
    
    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    if use_adversarial:
        # Use adversarial training
        trainer = AdversarialTrainer(model, optimizer, criterion, device)
        model, history = trainer.train(train_loader, val_loader, num_epochs=num_epochs, scheduler=scheduler)
    else:
        # Use standard training from train.py
        from train import train_model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device
        )
    
    # Save model
    model_name = f"{model_type}{'_adv' if use_adversarial else ''}_model.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as '{model_name}'")
    
    # Plot training history
    plot_training_history(history, model_type, use_adversarial)
    
    # Test robustness if adversarial training was used
    if use_adversarial:
        robustness = test_robustness(model, val_loader, criterion, device)
        print(f"Robustness metrics: {robustness}")
    
    return model, history

def plot_training_history(history, model_type, adversarial):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    if 'train_clean_loss' in history and 'train_adv_loss' in history:
        plt.plot(history['train_clean_loss'], label='Train (Clean)')
        plt.plot(history['train_adv_loss'], label='Train (Adv)')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_type}{'_adv' if adversarial else ''}_training_history.png")
    plt.close()
    print(f"Training history plot saved as '{model_type}{'_adv' if adversarial else ''}_training_history.png'")

def process_image_with_explanation(image_path, model_path, output_dir=None, model_type='standard', explanation_method='gradcam', device='cuda'):
    """Process an image with the enhanced model and generate explanations"""
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process image with appropriate model
    if model_type == 'hybrid':
        results = hybrid_process_image(image_path, model_path, output_dir, model_type, device)
    else:
        results = standard_process_image(image_path, model_path, output_dir, device)
    
    # Load model for explanation
    if model_type == 'hybrid':
        model = HybridForgeryDetector(num_classes=2, pretrained=False, with_segmentation=True)
    else:
        model = ForgeryDetectionNet(num_classes=2, pretrained=False, with_segmentation=True)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generate explanation
    if output_dir:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        explanation_path = os.path.join(output_dir, f"{name}_{explanation_method}_explanation.png")
    else:
        explanation_path = None
    
    # Generate explanation based on prediction
    target_class = results['prediction']  # Use the model's prediction as the target class
    explain_prediction(model, image_path, explanation_method, target_class, explanation_path, device)
    
    return results

def batch_process_with_explanation(image_dir, model_path, output_dir, model_type='standard', explanation_method='gradcam', device='cuda'):
    """Process all images in a directory with explanations"""
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image
    results = []
    for image_path in image_paths:
        print(f"Processing {os.path.basename(image_path)}...")
        result = process_image_with_explanation(
            image_path, model_path, output_dir, model_type, explanation_method, device
        )
        results.append((os.path.basename(image_path), result))
    
    # Create summary report
    authentic_count = sum(1 for _, r in results if r['prediction'] == 0)
    tampered_count = sum(1 for _, r in results if r['prediction'] == 1)
    
    print("\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"Authentic images: {authentic_count}")
    print(f"Tampered images: {tampered_count}")
    
    # Write detailed results to file
    with open(os.path.join(output_dir, 'enhanced_results.txt'), 'w') as f:
        f.write("Enhanced Image Forgery Detection Results\n")
        f.write("=====================================\n\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Explanation method: {explanation_method}\n\n")
        f.write(f"Total images processed: {len(results)}\n")
        f.write(f"Authentic images: {authentic_count}\n")
        f.write(f"Tampered images: {tampered_count}\n\n")
        f.write("Detailed Results:\n")
        f.write("---------------\n\n")
        
        for image_file, result in results:
            label = "Authentic" if result['prediction'] == 0 else "Tampered"
            confidence = result['prob_authentic'] if result['prediction'] == 0 else result['prob_tampered']
            f.write(f"Image: {image_file}\n")
            f.write(f"Prediction: {label}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
            f.write("\n")
    
    print(f"Detailed results saved to {os.path.join(output_dir, 'enhanced_results.txt')}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Image Forgery Detection')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'process', 'batch'], 
                        help='Mode: train a model, process a single image, or batch process images')
    parser.add_argument('--model_type', type=str, default='hybrid', choices=['standard', 'hybrid'], 
                        help='Type of model to use')
    parser.add_argument('--adversarial', action='store_true', 
                        help='Use adversarial training (only for train mode)')
    parser.add_argument('--image', type=str, 
                        help='Path to input image (for process mode)')
    parser.add_argument('--image_dir', type=str, 
                        help='Path to directory containing images (for batch mode)')
    parser.add_argument('--model', type=str, 
                        help='Path to trained model (for process and batch modes)')
    parser.add_argument('--output_dir', type=str, default='enhanced_results', 
                        help='Path to output directory')
    parser.add_argument('--explanation', type=str, default='gradcam', 
                        choices=['gradcam', 'integrated_gradients', 'lime'], 
                        help='Explanation method to use')
    parser.add_argument('--epochs', type=int, default=25, 
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train model
        train_enhanced_model(
            model_type=args.model_type,
            use_adversarial=args.adversarial,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    
    elif args.mode == 'process':
        # Process single image
        if not args.image or not args.model:
            parser.error("--image and --model are required for process mode")
        
        process_image_with_explanation(
            args.image,
            args.model,
            args.output_dir,
            args.model_type,
            args.explanation,
            args.device
        )
    
    elif args.mode == 'batch':
        # Process all images in directory
        if not args.image_dir or not args.model:
            parser.error("--image_dir and --model are required for batch mode")
        
        batch_process_with_explanation(
            args.image_dir,
            args.model,
            args.output_dir,
            args.model_type,
            args.explanation,
            args.device
        )

if __name__ == "__main__":
    main()