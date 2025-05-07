# Enhanced Image Forgery Detection System

This project implements an advanced image forgery detection system that combines deep learning with traditional forensic techniques to detect and localize image tampering. The system includes several novel components to improve detection accuracy, explainability, and robustness.

## Features

### 1. Hybrid Approach
Combines deep learning with traditional forensic features:
- **Deep Learning**: Uses a ResNet-based architecture with attention mechanisms
- **Traditional Forensic Features**: Incorporates Error Level Analysis (ELA), noise analysis, and texture features

### 2. Explainable AI
Provides visual explanations for model decisions using multiple techniques:
- **Grad-CAM**: Highlights regions that influenced the classification decision
- **Integrated Gradients**: Attributes prediction to input features
- **LIME**: Provides local interpretable explanations

### 3. Adversarial Training
Improves model robustness against adversarial attacks:
- Uses Projected Gradient Descent (PGD) to generate adversarial examples
- Trains on both clean and adversarial images
- Includes robustness evaluation metrics

### 4. Segmentation Capability
Locates tampered regions within an image:
- Generates pixel-level segmentation masks
- Provides visual overlays highlighting tampered areas

### 5. Attention Mechanism
Focuses on suspicious regions in the image:
- Uses multi-level attention maps
- Improves model interpretability

## Project Structure

- `model_architecture.py`: Base model architecture with attention mechanisms
- `forensic_features.py`: Traditional forensic feature extraction
- `hybrid_model.py`: Hybrid model combining deep learning and forensic features
- `adversarial_training.py`: Adversarial training implementation
- `explainable_ai.py`: Explanation methods for model decisions
- `inference.py`: Standard inference script
- `hybrid_inference.py`: Inference with hybrid model and forensic features
- `enhanced_forgery_detection.py`: Main script integrating all components
- `train.py`: Original training script

## Usage

### Training a Model

```bash
# Train a standard model
python enhanced_forgery_detection.py --mode train --model_type standard

# Train a hybrid model
python enhanced_forgery_detection.py --mode train --model_type hybrid

# Train with adversarial training
python enhanced_forgery_detection.py --mode train --model_type hybrid --adversarial
```

### Processing a Single Image

```bash
# Process with standard model
python enhanced_forgery_detection.py --mode process --model_type standard --model forgery_detection_model.pth --image path/to/image.jpg

# Process with hybrid model and Grad-CAM explanation
python enhanced_forgery_detection.py --mode process --model_type hybrid --model hybrid_model.pth --image path/to/image.jpg --explanation gradcam

# Process with different explanation method
python enhanced_forgery_detection.py --mode process --model_type hybrid --model hybrid_model.pth --image path/to/image.jpg --explanation integrated_gradients
```

### Batch Processing

```bash
# Process a directory of images
python enhanced_forgery_detection.py --mode batch --model_type hybrid --model hybrid_model.pth --image_dir path/to/images --output_dir results
```

## Results

The system generates several outputs for each processed image:

1. **Classification Result**: Authentic or tampered, with confidence score
2. **Segmentation Mask**: Highlighting tampered regions
3. **Attention Map**: Showing areas the model focused on
4. **Explanation Visualization**: Visual explanation of the model's decision
5. **Forensic Analysis**: For hybrid model, includes ELA visualization

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- opencv-python
- scikit-image
- scikit-learn

## Dataset

The system is designed to work with the CASIA dataset, which contains authentic and tampered images with corresponding masks.

## Future Work

- Real-time detection optimization
- Multi-modal fusion incorporating metadata analysis
- Integration with video forgery detection
- Deployment as a web service or desktop application
- Support for additional forensic techniques