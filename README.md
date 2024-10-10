# Deepfake Detection using State-of-the-Art Deep Learning Models

This project aims to improve the detection of deepfake videos using advanced deep learning techniques. The system integrates Long Short-Term Memory (LSTM) based Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) to detect and analyze both temporal and spatial anomalies in video frames. This approach provides a robust method to distinguish between authentic and manipulated media.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

## Introduction
With the rise in deepfake videos, this project addresses the need for robust automated systems that can effectively detect manipulated media. The project leverages state-of-the-art deep learning models to improve the detection accuracy and robustness across various datasets and use cases.

## Features
- **Deepfake Detection:** Identifies manipulated videos using advanced AI techniques.
- **Hybrid Model:** Combines CNN for spatial feature extraction and LSTM for temporal sequence analysis.
- **High Accuracy:** Achieves over 98% accuracy in detecting deepfakes, utilizing both frame-level and video-level analysis.
- **Extensive Dataset:** Trained on datasets like Face Forensics++, Deepfake Detection Challenge (DFDC), and Celeb-DF.

## Dataset
The system is trained using:
- **Face Forensics++**
- **Deepfake Detection Challenge (DFDC)**
- **Celeb-DF**

These datasets provide a balanced collection of real and manipulated videos to ensure the model generalizes well across different scenarios.

## Methodology
1. **Data Preprocessing:** 
   - Standardizes video resolution, frame rate, and aspect ratio.
   - Applies data augmentation techniques such as cropping, rotation, and flipping to improve model robustness.

2. **Model Architecture:**
   - **CNN** for spatial feature extraction from video frames.
   - **LSTM** for temporal sequence analysis of frame sequences to detect subtle anomalies.

3. **Training:** 
   - Utilizes the ResNext CNN model for feature extraction and a single LSTM layer for temporal analysis.
   - Hyperparameter tuning is applied to optimize model performance.

4. **Evaluation Metrics:**
   - Accuracy: 98.3%
   - Precision: 84.31%
   - Recall: 76.55%
   - F1 Score: 80.24%

## Model Architecture
- **CNN (Convolutional Neural Network):** Captures spatial details such as facial expressions and lighting.
- **LSTM (Long Short-Term Memory):** Analyzes temporal sequences to detect subtle anomalies across video frames.
  
![Model Architecture](path/to/model-architecture-image.png)

## Results
The model was trained on a large dataset, showing high accuracy in detecting deepfakes across varying numbers of frames per video.

| Number of Frames | Accuracy (%) |
|------------------|--------------|
| 10               | 84.21        |
| 20               | 87.79        |
| 40               | 89.35        |
| 60               | 90.59        |
| 80               | 91.50        |
| 100              | 98.35        |

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/deepfake-detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your dataset by downloading it from the [Face Forensics++](https://github.com/ondyari/FaceForensics) or [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge) repositories.
2. Run the training script:
    ```bash
    python train.py
    ```
3. For inference, use:
    ```bash
    python predict.py --video path/to/video
    ```

## Contributors
- **Vadrevu Venkata Sai Abhiram** (E21CSEU0673)
- **Vanshika Agrawal** (E21CSEU0684)
- **Yash Rathee** (E21CSEU0703)
- **Dhruv Chauhan** (E21CSEU0863)

## Acknowledgements
We extend our sincerest gratitude to **Dr. Rohit Kumar Kaliyar**, whose guidance and support were instrumental in the successful completion of this project.
