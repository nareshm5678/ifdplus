import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model_architecture import ForgeryDetectionNet
from hybrid_model import HybridForgeryDetector

class GradCAM:
    """Class for generating Grad-CAM visualizations for model interpretability"""
    
    def __init__(self, model, target_layer, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate class activation map"""
        # Forward pass
        if isinstance(self.model, HybridForgeryDetector):
            # For hybrid model, we need forensic features
            batch_size = input_tensor.size(0)
            forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
            outputs, _, _ = self.model(input_tensor, forensic_features)
        else:
            outputs, _, _ = self.model(input_tensor)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        
        # Backward pass
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weight activations by gradients
        cam = torch.sum(self.activations * gradients, dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.squeeze().cpu().numpy()

class IntegratedGradients:
    """Class for generating Integrated Gradients explanations"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def generate_integrated_gradients(self, input_tensor, target_class=None, steps=50):
        """Generate integrated gradients explanation"""
        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor).to(self.device)
        
        # Forward pass to get prediction
        if isinstance(self.model, HybridForgeryDetector):
            # For hybrid model, we need forensic features
            batch_size = input_tensor.size(0)
            forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
            outputs, _, _ = self.model(input_tensor, forensic_features)
        else:
            outputs, _, _ = self.model(input_tensor)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        
        # Generate scaled inputs
        scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        
        # Compute gradients for each scaled input
        gradients = []
        for i in range(0, len(scaled_inputs), 10):  # Process in batches to avoid memory issues
            batch = scaled_inputs[i:i+10]
            batch.requires_grad = True
            
            if isinstance(self.model, HybridForgeryDetector):
                batch_size = batch.size(0)
                forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
                batch_outputs, _, _ = self.model(batch, forensic_features)
            else:
                batch_outputs, _, _ = self.model(batch)
            
            # Target for backprop
            one_hot = torch.zeros_like(batch_outputs)
            one_hot[:, target_class] = 1
            
            # Backward pass
            self.model.zero_grad()
            batch_outputs.backward(gradient=one_hot, retain_graph=True)
            
            # Get gradients
            batch_gradients = batch.grad.detach()
            gradients.append(batch_gradients)
        
        # Concatenate gradients
        gradients = torch.cat(gradients, dim=0)
        
        # Average gradients
        avg_gradients = torch.mean(gradients, dim=0, keepdim=True)
        
        # Multiply by input - baseline
        integrated_gradients = (input_tensor - baseline) * avg_gradients
        
        # Sum over color channels
        attribution_map = torch.sum(integrated_gradients, dim=1, keepdim=True)
        
        # Normalize
        attribution_map = attribution_map - attribution_map.min()
        attribution_map = attribution_map / attribution_map.max() if attribution_map.max() > 0 else attribution_map
        
        return attribution_map.squeeze().cpu().numpy()

class LIME:
    """Simplified LIME implementation for image explanations"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def generate_lime_explanation(self, input_tensor, target_class=None, num_samples=1000, num_segments=50):
        """Generate LIME explanation"""
        # Convert tensor to numpy for segmentation
        input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Normalize for visualization
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        
        # Segment image
        segments = self._segment_image(input_np, num_segments)
        
        # Forward pass to get prediction
        if isinstance(self.model, HybridForgeryDetector):
            batch_size = input_tensor.size(0)
            forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
            outputs, _, _ = self.model(input_tensor, forensic_features)
        else:
            outputs, _, _ = self.model(input_tensor)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        
        # Generate perturbed samples
        perturbed_inputs, perturbation_mask = self._generate_samples(input_tensor, segments, num_samples)
        
        # Get predictions for perturbed samples
        predictions = []
        for i in range(0, len(perturbed_inputs), 10):  # Process in batches
            batch = perturbed_inputs[i:i+10]
            batch_tensor = torch.stack(batch).to(self.device)
            
            if isinstance(self.model, HybridForgeryDetector):
                batch_size = batch_tensor.size(0)
                forensic_features = torch.zeros(batch_size, 4, 256, 256).to(self.device)
                batch_outputs, _, _ = self.model(batch_tensor, forensic_features)
            else:
                batch_outputs, _, _ = self.model(batch_tensor)
            
            batch_probs = F.softmax(batch_outputs, dim=1)[:, target_class].detach().cpu().numpy()
            predictions.extend(batch_probs)
        
        # Fit linear model
        weights = self._fit_linear_model(perturbation_mask, predictions)
        
        # Generate explanation map
        explanation = np.zeros(segments.shape)
        for segment_id, weight in enumerate(weights):
            explanation[segments == segment_id] = weight
        
        # Normalize
        explanation = explanation - explanation.min()
        explanation = explanation / explanation.max() if explanation.max() > 0 else explanation
        
        return explanation
    
    def _segment_image(self, image, num_segments):
        """Segment image using SLIC"""
        try:
            from skimage.segmentation import slic
            segments = slic(image, n_segments=num_segments, compactness=10, start_label=0)
            return segments
        except ImportError:
            # Fallback to simple grid segmentation
            h, w = image.shape[:2]
            segments = np.zeros((h, w), dtype=np.int32)
            grid_size = int(np.sqrt(num_segments))
            segment_h, segment_w = h // grid_size, w // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    segments[i*segment_h:(i+1)*segment_h, j*segment_w:(j+1)*segment_w] = i * grid_size + j
            
            return segments
    
    def _generate_samples(self, input_tensor, segments, num_samples):
        """Generate perturbed samples by randomly turning segments on/off"""
        num_segments = np.max(segments) + 1
        perturbed_inputs = []
        perturbation_mask = []
        
        # Original input
        original = input_tensor.clone()
        
        # Create a baseline (black image)
        baseline = torch.zeros_like(original).to(self.device)
        
        for _ in range(num_samples):
            # Generate random binary mask
            mask = np.random.randint(0, 2, num_segments)
            perturbation_mask.append(mask)
            
            # Apply mask to image
            perturbed = original.clone()
            for segment_id in range(num_segments):
                if mask[segment_id] == 0:  # Turn off segment
                    segment_mask = (segments == segment_id)
                    perturbed[0, :, segment_mask] = baseline[0, :, segment_mask]
            
            perturbed_inputs.append(perturbed)
        
        return perturbed_inputs, np.array(perturbation_mask)
    
    def _fit_linear_model(self, perturbation_mask, predictions):
        """Fit a linear model to explain predictions"""
        from sklearn.linear_model import Ridge
        
        # Add bias term
        X = np.column_stack([perturbation_mask, np.ones(len(perturbation_mask))])
        
        # Fit ridge regression
        model = Ridge(alpha=1.0)
        model.fit(X, predictions)
        
        # Return coefficients (excluding bias)
        return model.coef_[:-1]

def visualize_explanation(image, explanation, method_name, output_path=None):
    """Visualize explanation overlay on image"""
    # Convert image to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * explanation), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Create overlay
    overlay = 0.7 * image + 0.3 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title(f'{method_name} Heatmap')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Explanation visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def explain_prediction(model, image_path, method='gradcam', target_class=None, output_path=None, device='cuda'):
    """Generate and visualize explanation for a prediction"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Original image for visualization
    original_image = np.array(image.resize((256, 256))) / 255.0
    
    # Generate explanation based on method
    if method.lower() == 'gradcam':
        # For ForgeryDetectionNet, use the last convolutional layer
        if isinstance(model, ForgeryDetectionNet):
            target_layer = model.backbone[7][-1].conv3  # Last conv layer in ResNet
        else:  # For HybridForgeryDetector
            target_layer = model.dl_branch.backbone[7][-1].conv3
        
        explainer = GradCAM(model, target_layer, device)
        explanation = explainer.generate_cam(image_tensor, target_class)
        method_name = 'Grad-CAM'
    
    elif method.lower() == 'integrated_gradients':
        explainer = IntegratedGradients(model, device)
        explanation = explainer.generate_integrated_gradients(image_tensor, target_class)
        method_name = 'Integrated Gradients'
    
    elif method.lower() == 'lime':
        explainer = LIME(model, device)
        explanation = explainer.generate_lime_explanation(image_tensor, target_class)
        method_name = 'LIME'
    
    else:
        raise ValueError(f"Unknown explanation method: {method}")
    
    # Visualize explanation
    visualize_explanation(original_image, explanation, method_name, output_path)
    
    return explanation

def explain_batch(model, image_paths, method='gradcam', output_dir=None, device='cuda'):
    """Generate explanations for a batch of images"""
    import os
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    explanations = []
    
    for image_path in image_paths:
        # Determine output path
        if output_dir:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}_{method}_explanation.png")
        else:
            output_path = None
        
        # Generate explanation
        explanation = explain_prediction(model, image_path, method, output_path=output_path, device=device)
        explanations.append(explanation)
    
    return explanations