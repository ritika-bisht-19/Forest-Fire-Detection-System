# Forest Fire Detection System

> An intelligent deep learning solution for real-time forest fire detection using Convolutional Neural Networks and advanced computer vision techniques.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [Author](#author)

---

## Overview

The Forest Fire Detection System leverages deep learning and CNN architecture to enable early detection of fires in forested areas. The system processes real-time video feeds, segments them into frames, and applies sophisticated feature engineering techniques to accurately classify fire conditions.

### Problem Statement
Forest fires pose significant threats to ecosystems, wildlife, and human settlements. Early detection is crucial for rapid response and mitigation. This system addresses the challenge by providing automated, real-time fire detection capabilities.

### Solution Approach
- **Deep Learning**: Utilizes transfer learning with InceptionV3 model
- **Computer Vision**: Advanced feature extraction and image processing
- **Real-time Processing**: Video feed segmentation and frame-by-frame analysis
- **Multi-class Classification**: Distinguishes between 'fire', 'no fire', and 'start fire' conditions

---

## Key Features

### Core Capabilities
- **Real-time Detection**: Processes live video feeds for immediate fire detection
- **Multi-class Classification**: Identifies fire, no fire, and early fire stages
- **Transfer Learning**: Leverages pre-trained InceptionV3 for efficient training
- **Feature Engineering**: Implements 8 advanced image processing techniques
- **High Accuracy**: Achieves 92% accuracy on test datasets

### Technical Highlights
- Frame-by-frame video analysis
- Comprehensive feature extraction pipeline
- Robust noise filtering and image enhancement
- Edge detection and keypoint identification
- Color space transformations for better fire detection

---

## Architecture

### Training Pipeline

```
Raw Images → Feature Engineering → CNN Model → Predictions → Evaluation
     ↓              ↓                 ↓           ↓          ↓
  Dataset    [8 Techniques]    InceptionV3   Fire/No Fire  Metrics
```

### Deployment Pipeline

```
Video Feed → Frame Extraction → Feature Engineering → Trained Model → Fire Detection
     ↓              ↓                   ↓                 ↓              ↓
  Real-time    Image Segments    [8 Techniques]      CNN Inference   Alert System
```

### Model Architecture Components

**Training Phase:**
1. **Data Input**: Labeled images (fire, no fire, start fire)
2. **Feature Engineering**: Multi-technique feature extraction
3. **Model Training**: Transfer learning with InceptionV3
4. **Prediction**: Probability-based classification
5. **Evaluation**: Performance metrics and validation
6. **Hyperparameter Tuning**: Optimization using validation sets

**Deployment Phase:**
1. **Video Input**: Real-time video feed
2. **Frame Segmentation**: Video-to-image conversion
3. **Feature Processing**: Same techniques as training
4. **Inference**: Fire detection using trained model
5. **Alert Generation**: Real-time fire detection alerts

---

## Feature Engineering

Our system implements 8 sophisticated feature engineering techniques to enhance fire detection accuracy:

### 1. Noise Filtering
- **Gaussian Filter**: Low-pass filtering for noise reduction
- **Bilateral Filter**: Edge-preserving smoothing
- **Median Filter**: Random noise reduction while maintaining edges

### 2. Image Segmentation
- **Otsu's Thresholding**: Automatic threshold-based segmentation
- **Pixel Classification**: Boundary detection and region identification
- **Histogram Analysis**: Peak value detection for optimal thresholding

### 3. Color Space Transformation
- **RGB to LAB**: Enhanced color spectrum representation
- **Lightness Component**: 0-100 range lightness analysis
- **Color Channels**: Red-green and yellow-blue traversal for fire detection

### 4. Thresholding Techniques
- **Binary Thresholding**: Output binary images for analysis
- **Adaptive Thresholding**: Gaussian and mean adaptive methods
- **Fire/Smoke Segmentation**: Background separation for better analysis

### 5. Edge Detection
- **Sobel Filters**: Horizontal and vertical edge detection
- **Brightness Discontinuity**: Intensity change identification
- **Fire Boundary Detection**: Sharp intensity transitions in fire regions

### 6. ORB Keypoint Detection
- **FAST Algorithm**: Features from Accelerated Segment Testing
- **BRIEF Descriptors**: Binary Robust Independent Elementary Features
- **Corner Detection**: Sharp pixel intensity shifts
- **Feature Matching**: Keypoint descriptor comparison

### 7. Image Orientation
- **Angle Correction**: Proper image alignment for consistency
- **Pattern Recognition**: Consistent class categorization
- **Real-time Adaptation**: Handling various camera angles

### 8. Censure and Corner Detection
- **Feature Detection**: Significant feature identification
- **Harris Corner Detector**: Corner localization and feature inference
- **Real-time Processing**: Efficient feature extraction for live feeds

---

## Model Performance

### Experimental Results

#### Combination 1: Gaussian + Segmentation + LAB
- **Accuracy**: 77.23%
- **Precision**: 70%
- **Recall**: 71%
- **Strengths**: Good performance across all three classes
- **Test Results**: 286 fire, 224 no-fire, 26 start-fire correctly classified

#### Combination 2: Edge Detection + Gaussian + ORB
- **Accuracy**: 77%
- **Precision**: 51%
- **Recall**: 55%
- **Strengths**: Excellent fire/no-fire classification
- **Weakness**: Poor start-fire detection
- **Test Results**: 283 fire, 252 no-fire correctly classified

#### Final Model: All Features Combined
- **Accuracy**: 92%
- **Precision**: 90%
- **Recall**: 81%
- **Test Results**: 309 fire, 299 no-fire, 25 start-fire correctly classified
- **Total Test Samples**: 694

### Performance Metrics
- **Confusion Matrix**: Diagonal dominance indicating strong classification
- **ROC Curves**: Excellent area under curve values
- **Cross-validation**: Consistent performance across different data splits

---

## Implementation Details

### Technical Stack
- **Framework**: TensorFlow/Keras
- **Base Model**: InceptionV3 (Transfer Learning)
- **Image Processing**: OpenCV, scikit-image
- **Feature Extraction**: Custom pipeline with 8 techniques
- **Video Processing**: Frame-by-frame segmentation

### Training Configuration
- **Epochs**: 50 (optimized through experimentation)
- **Batch Size**: Tuned for optimal performance
- **Learning Rate**: Adaptive learning rate scheduling
- **Regularization**: Dropout and batch normalization
- **Data Augmentation**: Rotation, scaling, and flipping

### Hyperparameter Optimization
- **Validation Strategy**: Hold-out validation for tuning
- **Grid Search**: Systematic parameter exploration
- **Performance Monitoring**: Real-time accuracy and loss tracking
- **Early Stopping**: Preventing overfitting

---

## Results

### Key Achievements
- **High Accuracy**: 92% on unseen test data
- **Real-time Processing**: Efficient video feed analysis
- **Multi-class Detection**: Accurate fire stage classification
- **Robust Performance**: Consistent results across various scenarios

### Validation Results
- **Training Accuracy**: Steady improvement over epochs
- **Validation Accuracy**: Strong generalization capability
- **Loss Curves**: Consistent decrease indicating proper learning
- **Confusion Matrix**: High diagonal values showing accurate classification

### Real-world Testing
- **Video Feed Analysis**: Successful real-time fire detection
- **Frame Processing**: Efficient image segmentation and analysis
- **Alert System**: Timely fire detection notifications
- **Scalability**: Capable of handling multiple video streams

---

## Usage

### Prerequisites
```bash
pip install tensorflow opencv-python scikit-image numpy matplotlib
```

### Quick Start
```python
# Load the trained model
model = load_model('forest_fire_model.h5')

# Process video feed
python image_capture.py --input video.mp4 --output frames/

# Run detection
python fire_detection.py --model forest_fire_model.h5 --input frames/
```

### Video Processing
The [image_capture.py](image_capture.py) script segments videos into individual frames for analysis:
```bash
python image_capture.py --video_path input.mp4 --output_dir frames/
```

---

## Contributing

We welcome contributions to improve the Forest Fire Detection System. Please follow these guidelines:

1. **Fork the Repository**: Create your feature branch
2. **Code Standards**: Follow PEP 8 for Python code
3. **Testing**: Ensure all tests pass before submitting
4. **Documentation**: Update relevant documentation
5. **Pull Request**: Submit with detailed description

### Development Setup
```bash
git clone https://github.com/savetree-1/Forest-Fire-Detection-System.git
cd Forest-Fire-Detection-System
pip install -r requirements.txt
```

---

## Author

**Ankush Rawat**
- GitHub: [@savetree-1](https://github.com/savetree-1)
- Project: [Forest Fire Detection System](https://github.com/savetree-1/Forest-Fire-Detection-System)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Transfer learning approach using InceptionV3
- Advanced computer vision techniques for feature extraction
- Open-source libraries and frameworks that made this project possible
- Research community contributions in fire detection and computer vision

---

*Built with passion for environmental protection and cutting-edge technology.*