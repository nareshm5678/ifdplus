import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model_architecture import ForgeryDetectionNet, ForgeryDetectionLoss
from hybrid_model import HybridForgeryDetector, HybridForgeryDetectionLoss

class AdversarialTrainer:
    """Class for adversarial training of forgery detection models"""
    
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def generate_adversarial_examples(self, inputs, labels, masks=None, epsilon=0.03, alpha=0.01, iterations=10):
        """Generate adversarial examples using Projected Gradient Descent (PGD)"""
        # Clone the inputs to avoid modifying them
        adv_inputs = inputs.clone().detach().to(self.device)
        labels = labels.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        
        # Add small random noise to start from a different point
        adv_inputs = adv_inputs + torch.empty_like(adv_inputs).uniform_(-epsilon, epsilon)
        adv_inputs = torch.clamp(adv_inputs, 0, 1)  # Ensure valid image range
        
        # PGD iterations
        for _ in range(iterations):
            adv_inputs.requires_grad = True
            
            # Forward pass
            if isinstance(self.model, HybridForgeryDetector):
                # For hybrid model, we need forensic features
                # In practice, these would be extracted from the adversarial images
                # Here we use zeros as a placeholder
                batch_size = inputs.size(0)
                forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
                outputs, segmentations, attention_maps = self.model(adv_inputs, forensic_features)
            else:
                outputs, segmentations, attention_maps = self.model(adv_inputs)
            
            # Calculate loss
            if masks is not None:
                loss = self.criterion(outputs, labels, segmentations, masks, attention_maps)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                # Move in the direction of increasing loss
                adv_inputs = adv_inputs + alpha * adv_inputs.grad.sign()
                
                # Project back to epsilon ball
                delta = torch.clamp(adv_inputs - inputs, -epsilon, epsilon)
                adv_inputs = torch.clamp(inputs + delta, 0, 1)
            
            # Reset gradients
            self.optimizer.zero_grad()
        
        return adv_inputs
    
    def adversarial_training_step(self, inputs, labels, masks=None, epsilon=0.03):
        """Perform one step of adversarial training"""
        # Generate adversarial examples
        adv_inputs = self.generate_adversarial_examples(inputs, labels, masks, epsilon)
        
        # Train on both clean and adversarial examples
        self.optimizer.zero_grad()
        
        # Forward pass on clean examples
        if isinstance(self.model, HybridForgeryDetector):
            # For hybrid model, we need forensic features
            batch_size = inputs.size(0)
            forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
            clean_outputs, clean_segmentations, clean_attention_maps = self.model(inputs, forensic_features)
        else:
            clean_outputs, clean_segmentations, clean_attention_maps = self.model(inputs)
        
        # Calculate loss on clean examples
        if masks is not None:
            clean_loss = self.criterion(clean_outputs, labels, clean_segmentations, masks, clean_attention_maps)
        else:
            clean_loss = self.criterion(clean_outputs, labels)
        
        # Forward pass on adversarial examples
        if isinstance(self.model, HybridForgeryDetector):
            adv_outputs, adv_segmentations, adv_attention_maps = self.model(adv_inputs, forensic_features)
        else:
            adv_outputs, adv_segmentations, adv_attention_maps = self.model(adv_inputs)
        
        # Calculate loss on adversarial examples
        if masks is not None:
            adv_loss = self.criterion(adv_outputs, labels, adv_segmentations, masks, adv_attention_maps)
        else:
            adv_loss = self.criterion(adv_outputs, labels)
        
        # Combined loss
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        # Backward pass and optimize
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), clean_loss.item(), adv_loss.item()
    
    def train_epoch(self, train_loader, epsilon=0.03):
        """Train for one epoch with adversarial examples"""
        self.model.train()
        running_loss = 0.0
        running_clean_loss = 0.0
        running_adv_loss = 0.0
        
        for inputs, masks, labels in train_loader:
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Perform adversarial training step
            loss, clean_loss, adv_loss = self.adversarial_training_step(inputs, labels, masks, epsilon)
            
            # Update statistics
            running_loss += loss * inputs.size(0)
            running_clean_loss += clean_loss * inputs.size(0)
            running_adv_loss += adv_loss * inputs.size(0)
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_clean_loss = running_clean_loss / len(train_loader.dataset)
        epoch_adv_loss = running_adv_loss / len(train_loader.dataset)
        
        return epoch_loss, epoch_clean_loss, epoch_adv_loss
    
    def evaluate(self, val_loader):
        """Evaluate the model on validation data"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for inputs, masks, labels in val_loader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if isinstance(self.model, HybridForgeryDetector):
                    # For hybrid model, we need forensic features
                    batch_size = inputs.size(0)
                    forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
                    outputs, segmentations, attention_maps = self.model(inputs, forensic_features)
                else:
                    outputs, segmentations, attention_maps = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, labels, segmentations, masks, attention_maps)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct.double() / len(val_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, train_loader, val_loader, num_epochs=10, epsilon=0.03, scheduler=None):
        """Train the model with adversarial examples"""
        best_val_acc = 0.0
        best_model_wts = self.model.state_dict().copy()
        
        history = {
            'train_loss': [], 'train_clean_loss': [], 'train_adv_loss': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Train with adversarial examples
            train_loss, train_clean_loss, train_adv_loss = self.train_epoch(train_loader, epsilon)
            
            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update learning rate if scheduler is provided
            if scheduler is not None:
                scheduler.step()
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f} (Clean: {train_clean_loss:.4f}, Adv: {train_adv_loss:.4f})')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_clean_loss'].append(train_clean_loss)
            history['train_adv_loss'].append(train_adv_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = self.model.state_dict().copy()
            
            print()
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        return self.model, history

# Function to create an adversarial trainer
def create_adversarial_trainer(model, optimizer, criterion, device='cuda'):
    """Create an adversarial trainer for a forgery detection model"""
    return AdversarialTrainer(model, optimizer, criterion, device)

# Function to test model robustness against adversarial attacks
def test_robustness(model, test_loader, criterion, device='cuda', epsilon=0.03):
    """Test model robustness against adversarial attacks"""
    model.eval()
    
    # Create a temporary optimizer for generating adversarial examples
    temp_optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create an adversarial trainer for generating adversarial examples
    adv_trainer = AdversarialTrainer(model, temp_optimizer, criterion, device)
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for inputs, masks, labels in test_loader:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        # Generate adversarial examples
        adv_inputs = adv_trainer.generate_adversarial_examples(inputs, labels, masks, epsilon)
        
        # Evaluate on clean examples
        with torch.no_grad():
            if isinstance(model, HybridForgeryDetector):
                # For hybrid model, we need forensic features
                batch_size = inputs.size(0)
                forensic_features = torch.zeros(batch_size, 4, 256, 256).to(device)
                clean_outputs, _, _ = model(inputs, forensic_features)
                adv_outputs, _, _ = model(adv_inputs, forensic_features)
            else:
                clean_outputs, _, _ = model(inputs)
                adv_outputs, _, _ = model(adv_inputs)
            
            # Calculate accuracy
            _, clean_preds = torch.max(clean_outputs, 1)
            _, adv_preds = torch.max(adv_outputs, 1)
            
            clean_correct += torch.sum(clean_preds == labels.data)
            adv_correct += torch.sum(adv_preds == labels.data)
            total += labels.size(0)
    
    # Calculate accuracy
    clean_acc = clean_correct.double() / total
    adv_acc = adv_correct.double() / total
    
    print(f'Clean Accuracy: {clean_acc:.4f}')
    print(f'Adversarial Accuracy: {adv_acc:.4f}')
    print(f'Robustness Gap: {clean_acc - adv_acc:.4f}')
    
    return {
        'clean_acc': clean_acc.item(),
        'adv_acc': adv_acc.item(),
        'robustness_gap': (clean_acc - adv_acc).item()
    }