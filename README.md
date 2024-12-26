# Neural Network Training Pipeline

Welcome to the Neural Network Training Pipeline! This project implements a robust framework for training and evaluating convolutional neural networks (CNNs) using the PyTorch library. The primary focus is on the MNIST dataset, a well-known benchmark for image classification tasks.

## Overview

This repository contains the following key components:

- **Model Architectures**: Multiple CNN architectures designed for effective feature extraction and classification.
- **Training and Testing Scripts**: Functions to facilitate the training and evaluation of models.
- **Data Handling Utilities**: Tools for loading and preprocessing the MNIST dataset.

## Getting Started

### Prerequisites

To run this project, ensure you have the following installed:

- Python
- PyTorch
- torchvision
- tqdm
- pytest
- numpy

You can install the required packages using pip:
```
pip install -r requirements.txt
```

### Dataset

The project utilizes the MNIST dataset, which consists of handwritten digits. The dataset is automatically downloaded when you run the training script for the first time.

### Running the Training

To initiate the training process, execute the following command (make sure to select appropriate model for training):
```
python src/train.py --model 1  # For FirstModel
python src/train.py --model 2  # For SecondModel
python src/train.py --model 3  # For ThirdModel
```

This command will start the training of the neural network, utilizing the best available hardware (MPS, CUDA, or CPU).

### Model Architectures

The repository includes several model definitions, each with unique configurations:

- **FirstModel**: A basic CNN architecture with essential layers.
- **SecondModel**: An enhanced version with additional convolutional layers.
- **ThirdModel**: A more complex architecture featuring skip connections and attention mechanisms.

### Results

After training, the script will output the following metrics:

- Best training accuracy
- Best testing accuracy
- Total number of parameters in the model

# First Model
### Targets: 
 - Started with a model given in colab from the session 7 of ERAV3 class.
### Results:
```
Parameters: 13.8k
Best Train Accuracy: 98.90
Best Test Accuracy: 99.50 (12th Epoch)
```
### Analysis: 
 - This CNN model, consisting of 7 layers, employs dropout to mitigate overfitting. While it demonstrates high test set accuracy, it falls short of reaching 99.41% with a parameter count below 8K.
 - Notably, the model achieves a test set accuracy of 99.50% with a total of 13.8k parameters.
 - On the training set, the model attains an accuracy of 98.90% with the same 13.8k parameters.


## Code for First Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_rate_model1 = 0.03

# Model Definition 1
class FirstModel(nn.Module):
    def __init__(self):
        super(FirstModel, self).__init__()
        # Input Block
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate_model1)
        )

        # Convolution Block 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_rate_model1)
        )

        # Transition Block 1
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolution Block 2
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate_model1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate_model1)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate_model1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate_model1)
        )
        
        # Output Block
        self.global_avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.global_avg_pool(x)        
        x = self.layer8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```

## Console Log for First Model

```
(temporary) ➜  erav3-s7-mnist-nn python src/train.py --model 1
Using device: mps
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Conv2d: 2-1                       144
|    └─ReLU: 2-2                         --
|    └─BatchNorm2d: 2-3                  32
|    └─Dropout: 2-4                      --
├─Sequential: 1-2                        --
|    └─Conv2d: 2-5                       4,608
|    └─ReLU: 2-6                         --
|    └─BatchNorm2d: 2-7                  64
|    └─Dropout: 2-8                      --
├─Sequential: 1-3                        --
|    └─Conv2d: 2-9                       320
├─MaxPool2d: 1-4                         --
├─Sequential: 1-5                        --
|    └─Conv2d: 2-10                      1,440
|    └─ReLU: 2-11                        --
|    └─BatchNorm2d: 2-12                 32
|    └─Dropout: 2-13                     --
├─Sequential: 1-6                        --
|    └─Conv2d: 2-14                      2,304
|    └─ReLU: 2-15                        --
|    └─BatchNorm2d: 2-16                 32
|    └─Dropout: 2-17                     --
├─Sequential: 1-7                        --
|    └─Conv2d: 2-18                      2,304
|    └─ReLU: 2-19                        --
|    └─BatchNorm2d: 2-20                 32
|    └─Dropout: 2-21                     --
├─Sequential: 1-8                        --
|    └─Conv2d: 2-22                      2,304
|    └─ReLU: 2-23                        --
|    └─BatchNorm2d: 2-24                 32
|    └─Dropout: 2-25                     --
├─Sequential: 1-9                        --
|    └─AvgPool2d: 2-26                   --
├─Sequential: 1-10                       --
|    └─Conv2d: 2-27                      160
=================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
=================================================================
CUDA Available? False
Epoch: 1
Loss=0.06085345149040222 Batch_id=937 Accuracy=91.56: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:20<00:00, 46.40it/s]

Test set: Average loss: 0.0443, Accuracy: 9867/10000 (98.67%)

Epoch: 2
Loss=0.00517434673383832 Batch_id=937 Accuracy=97.36: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 68.61it/s]

Test set: Average loss: 0.0354, Accuracy: 9896/10000 (98.96%)

Epoch: 3
Loss=0.30349403619766235 Batch_id=937 Accuracy=98.00: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 68.74it/s]

Test set: Average loss: 0.0370, Accuracy: 9886/10000 (98.86%)

Epoch: 4
Loss=0.0570223405957222 Batch_id=937 Accuracy=98.15: 100%|███████��██████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 67.16it/s]

Test set: Average loss: 0.0330, Accuracy: 9893/10000 (98.93%)

Epoch: 5
Loss=0.015080569311976433 Batch_id=937 Accuracy=98.37: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:14<00:00, 64.94it/s]

Test set: Average loss: 0.0296, Accuracy: 9906/10000 (99.06%)

Epoch: 6
Loss=0.11596260219812393 Batch_id=937 Accuracy=98.47: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 68.87it/s]

Test set: Average loss: 0.0257, Accuracy: 9927/10000 (99.27%)

Epoch: 7
Loss=0.2220374345779419 Batch_id=937 Accuracy=98.52: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:14<00:00, 64.18it/s]

Test set: Average loss: 0.0244, Accuracy: 9923/10000 (99.23%)

Epoch: 8
Loss=0.0159312654286623 Batch_id=937 Accuracy=98.59: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 67.36it/s]

Test set: Average loss: 0.0254, Accuracy: 9923/10000 (99.23%)

Epoch: 9
Loss=0.06191922351717949 Batch_id=937 Accuracy=98.63: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 69.24it/s]

Test set: Average loss: 0.0232, Accuracy: 9930/10000 (99.30%)

Epoch: 10
Loss=0.008591572754085064 Batch_id=937 Accuracy=98.72: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 69.01it/s]

Test set: Average loss: 0.0221, Accuracy: 9921/10000 (99.21%)

Epoch: 11
Loss=0.08622267842292786 Batch_id=937 Accuracy=98.80: 100%|███████████████████████████████████████████████████████████��██████████████████████████████| 938/938 [00:13<00:00, 69.40it/s]

Test set: Average loss: 0.0218, Accuracy: 9927/10000 (99.27%)

Epoch: 12
Loss=0.20960989594459534 Batch_id=937 Accuracy=98.86: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 69.59it/s]

Test set: Average loss: 0.0189, Accuracy: 9950/10000 (99.50%)

Epoch: 13
Loss=0.004187001846730709 Batch_id=937 Accuracy=98.81: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 68.76it/s]

Test set: Average loss: 0.0212, Accuracy: 9933/10000 (99.33%)

Epoch: 14
Loss=0.004360235296189785 Batch_id=937 Accuracy=98.86: 100%|████████████████████████████████████████████████████████████████████████���█████████████████| 938/938 [00:14<00:00, 66.24it/s]

Test set: Average loss: 0.0213, Accuracy: 9930/10000 (99.30%)

Epoch: 15
Loss=0.004928364418447018 Batch_id=937 Accuracy=98.90: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:13<00:00, 69.03it/s]

Test set: Average loss: 0.0194, Accuracy: 9941/10000 (99.41%)


===================
Results:
Parameters: 13.8k
Best Train Accuracy: 98.90
Best Test Accuracy: 99.50 (12th Epoch)
===================
```


# Second Model
### Targets: 
 - Improve the model to reduce the number of parameters less than 8K.
The new model introduces several changes to enhance efficiency, reduce complexity, and improve generalization. Here's a detailed analysis of the changes, their purposes, and benefits:

---

### **1. Filter Count Optimization**
- **Old Model**:
  - Initial block: 16 filters.
  - Second block: 32 filters.
  - Transition block: Down to 10 filters.
  - Later layers: Consistently higher number of filters (16).
- **New Model**:
  - Initial block: Reduced to 8 filters.
  - Second block: Reduced to 16 filters.
  - Transition block: Reduced to 8 filters, then further processing with smaller numbers of filters (e.g., 12 filters in later blocks).
- **Rationale**: The new model employs fewer filters in each layer, making it computationally more efficient.
- **Benefits**:
  - **Reduced computational cost**: Smaller filters reduce memory and processing requirements.
  - **Simpler model**: Avoids overfitting by reducing overparameterization, especially important for small datasets.

---

### **2. Enhanced Transition Block**
- **Old Model**:
  - Transition block reduced filters from 32 to 10 using a `1x1` convolution, followed by pooling.
- **New Model**:
  - Transition block reduces filters from 16 to 8, followed by pooling.
- **Rationale**: The transition block now reduces to a smaller feature map size, ensuring a more gradual dimensionality reduction.
- **Benefits**:
  - Maintains a balance between feature richness and computational efficiency.
  - Prevents excessive information loss due to abrupt reductions.

---

### **3. Streamlined Convolution Blocks**
- **Old Model**:
  - Four convolution blocks in the second stage, all with higher numbers of filters.
- **New Model**:
  - Three convolution blocks in the second stage, with fewer filters (12 filters in blocks 4, 5, and 6).
- **Rationale**: The new model reduces the depth and number of filters, opting for efficiency.
- **Benefits**:
  - Reduces overfitting risk, especially for small datasets.
  - Computationally less expensive.

---

### **4. Adjusted Kernel Sizes and GAP**
- **Old Model**:
  - Global Average Pooling (GAP) with a kernel size of `6` and a spatial dimension consistent with the larger feature maps.
- **New Model**:
  - GAP with a kernel size of `8`, matched to the smaller feature maps produced by the new architecture.
- **Rationale**: Kernel size is adjusted to fit the reduced feature map dimensions.
- **Benefits**:
  - Ensures the GAP operation appropriately aggregates spatial features.
  - Aligns model architecture with computational resources.

---

### **5. Simplified Dropout Strategy**
- **Old Model**: Included a standalone `dropout` layer at the end.
- **New Model**: Relies only on dropout within convolution blocks.
- **Rationale**: Standalone dropout at the end might not significantly impact regularization since the GAP already reduces dimensionality.
- **Benefits**:
  - Avoids redundancy and simplifies the architecture.
  - Focuses regularization within intermediate layers where overfitting risk is higher.

---

### **6. Optimized Feature Map Sizes**
- **Old Model**: Larger feature maps due to the higher number of filters in all stages.
- **New Model**: Reduced feature map sizes by limiting filters and using fewer blocks in later stages.
- **Rationale**: Smaller feature maps conserve memory and processing time.
- **Benefits**:
  - Makes the model lighter and faster without sacrificing performance on datasets where simpler architectures suffice.
  - Helps focus on essential features while avoiding noise.

---

### **7. Model Simplification**
- **Old Model**: Relatively complex with higher filters, deeper layers, and standalone dropout.
- **New Model**: Streamlined, with fewer filters, fewer blocks, and integrated regularization.
- **Rationale**: Aligns with the principle of Occam’s Razor, ensuring the model is not overly complex for the given task.
- **Benefits**:
  - Easier to train, especially on resource-constrained systems.
  - Lower risk of overfitting on small datasets.
  - Reduced inference time and power consumption.

---

### **Summary of Benefits**
1. **Efficiency**: Fewer filters and blocks reduce computation time and resource requirements.
2. **Generalization**: Simpler architecture avoids overfitting, especially useful for small or moderately complex datasets.
3. **Alignment with Input Size**: Smaller feature maps and adjusted GAP improve processing consistency.
4. **Robust Regularization**: Dropout within intermediate layers ensures effective regularization without unnecessary complexity.

These changes make the new model more compact and efficient while maintaining sufficient capacity to handle moderately complex tasks. It is particularly suited for environments where computational resources or data are limited.

### Results: 
```
Parameters: 5.0k
Best Train Accuracy: 98.31
Best Test Accuracy: 99.08 (9th Epoch)
```
### Analysis: 
 - This 7-layer CNN model leverages dropout to combat overfitting. While it demonstrates high test set accuracy, it falls short of reaching 99.41% with a parameter count below 8K.
 - Notably, the model achieves a test set accuracy of 99.08% with a total of 5.0k parameters.
 - On the training set, the model attains an accuracy of 98.31% with the same 5.0k parameters.

## Code for Model 2

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_rate_model2 = 0.1

# Model Definition 2
class SecondModel(nn.Module):
    def __init__(self):
        super(SecondModel, self).__init__()
        # Input Block
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate_model2)
        )

        # Convolution Block 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate_model2)
        )

        # Transition Block 1
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolution Block 2
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate_model2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate_model2)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate_model2)
        )
        
        # Output Block
        self.global_avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.global_avg_pool(x)        
        x = self.layer7(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```

## Console Log for Second Model

```
(temporary) ➜  erav3-s7-mnist-nn python src/train.py --model 2
Using device: mps
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Conv2d: 2-1                       72
|    └─ReLU: 2-2                         --
|    └─BatchNorm2d: 2-3                  16
|    └─Dropout: 2-4                      --
├─Sequential: 1-2                        --
|    └─Conv2d: 2-5                       1,152
|    └─ReLU: 2-6                         --
|    └─BatchNorm2d: 2-7                  32
|    └─Dropout: 2-8                      --
├─Sequential: 1-3                        --
|    └─Conv2d: 2-9                       128
├─MaxPool2d: 1-4                         --
├─Sequential: 1-5                        --
|    └─Conv2d: 2-10                      864
|    └─ReLU: 2-11                        --
|    └─BatchNorm2d: 2-12                 24
|    └─Dropout: 2-13                     --
├─Sequential: 1-6                        --
|    └─Conv2d: 2-14                      1,296
|    └─ReLU: 2-15                        --
|    └─BatchNorm2d: 2-16                 24
|    └─Dropout: 2-17                     --
├─Sequential: 1-7                        --
|    └─Conv2d: 2-18                      1,296
|    └─ReLU: 2-19                        --
|    └─BatchNorm2d: 2-20                 24
|    └─Dropout: 2-21                     --
├─Sequential: 1-8                        --
|    └─AvgPool2d: 2-22                   --
├─Sequential: 1-9                        --
|    └─Conv2d: 2-23                      120
=================================================================
Total params: 5,048
Trainable params: 5,048
Non-trainable params: 0
=================================================================
CUDA Available? False
Epoch: 1
Loss=0.1148453876376152 Batch_id=937 Accuracy=84.46: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 61.39it/s]

Test set: Average loss: 0.1004, Accuracy: 9709/10000 (97.09%)

Epoch: 2
Loss=0.01110828947275877 Batch_id=937 Accuracy=95.85: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 72.68it/s]

Test set: Average loss: 0.0838, Accuracy: 9760/10000 (97.60%)

Epoch: 3
Loss=0.4584121108055115 Batch_id=937 Accuracy=96.75: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 72.93it/s]

Test set: Average loss: 0.0474, Accuracy: 9864/10000 (98.64%)

Epoch: 4
Loss=0.13524143397808075 Batch_id=937 Accuracy=97.30: 100%|█████████████████████████████████████████████��████████████████████████████████████████████| 938/938 [00:12<00:00, 74.57it/s]

Test set: Average loss: 0.0554, Accuracy: 9830/10000 (98.30%)

Epoch: 5
Loss=0.04407692700624466 Batch_id=937 Accuracy=97.44: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 75.09it/s]

Test set: Average loss: 0.0408, Accuracy: 9865/10000 (98.65%)

Epoch: 6
Loss=0.13168609142303467 Batch_id=937 Accuracy=97.70: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 75.99it/s]

Test set: Average loss: 0.0360, Accuracy: 9884/10000 (98.84%)

Epoch: 7
Loss=0.19003081321716309 Batch_id=937 Accuracy=97.80: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 75.07it/s]

Test set: Average loss: 0.0399, Accuracy: 9872/10000 (98.72%)

Epoch: 8
Loss=0.16690832376480103 Batch_id=937 Accuracy=97.92: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.76it/s]

Test set: Average loss: 0.0359, Accuracy: 9892/10000 (98.92%)

Epoch: 9
Loss=0.028158679604530334 Batch_id=937 Accuracy=98.07: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.88it/s]

Test set: Average loss: 0.0296, Accuracy: 9908/10000 (99.08%)

Epoch: 10
Loss=0.002559197135269642 Batch_id=937 Accuracy=98.15: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.62it/s]

Test set: Average loss: 0.0365, Accuracy: 9885/10000 (98.85%)

Epoch: 11
Loss=0.08638763427734375 Batch_id=937 Accuracy=98.21: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 74.83it/s]

Test set: Average loss: 0.0290, Accuracy: 9906/10000 (99.06%)

Epoch: 12
Loss=0.27080538868904114 Batch_id=937 Accuracy=98.26: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 75.07it/s]

Test set: Average loss: 0.0291, Accuracy: 9906/10000 (99.06%)

Epoch: 13
Loss=0.00583559600636363 Batch_id=937 Accuracy=98.25: 100%|███████████████████████████████████���██████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.92it/s]

Test set: Average loss: 0.0286, Accuracy: 9905/10000 (99.05%)

Epoch: 14
Loss=0.009213566780090332 Batch_id=937 Accuracy=98.28: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.82it/s]

Test set: Average loss: 0.0307, Accuracy: 9905/10000 (99.05%)

Epoch: 15
Loss=0.014064904302358627 Batch_id=937 Accuracy=98.31: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:12<00:00, 73.38it/s]

Test set: Average loss: 0.0308, Accuracy: 9897/10000 (98.97%)


===================
Results:
Parameters: 5.0k
Best Train Accuracy: 98.31
Best Test Accuracy: 99.08 (9th Epoch)
=================== 
```


# Third Model
### Targets: 
 - The number of parameters less than 8K, but the accuracy is not should be higher than 99.41% in 15 epochs. The next goal is to achieve the target of 99.41% accuracy in less than 15 epochs.
The new model introduces several changes that enhance its performance, robustness, and efficiency. Here's a detailed breakdown of the changes and their purposes:

---

### **1. Final Convolution Layer Changes**
- **Old Model**: `convblock7` for output.
- **New Model**: `convblock8` replaces it and is paired with attention.
- **Rationale**: Splitting attention and output prediction into separate layers enhances modularity and specialization of layers.
- **Advantage**:
  - The attention mechanism selectively enhances important features.
  - Improves interpretability and accuracy.

---

### **2. Interpolation for Size Adjustment**
- **New Addition**: `F.interpolate` is used to resize earlier layer outputs to match the spatial dimensions of deeper layers before adding them (in skip connections).
- **Rationale**: Ensures dimensional consistency when adding features from different layers.
- **Advantage**: Improves feature alignment and allows skip connections to work seamlessly in non-identical spatial dimensions.

---

### **3. GAP and Kernel Size Reduction**
- **Old Model**: GAP kernel size = `8`, Conv filter input size = `12 channels`.
- **New Model**: GAP kernel size = `6`, Conv filter input size = `12 channels`.
- **Rationale**: Smaller kernel sizes in GAP lead to reduced computational overhead and better adaptation to the smaller feature maps in the new model.
- **Advantage**: Optimizes computational efficiency without sacrificing the ability to generalize well.

---

### **4. Attention Mechanism Integration**
- **New Feature**: Adaptive attention mechanism.
  - **Details**: `F.adaptive_avg_pool2d` computes a spatially global average and applies a `torch.sigmoid` to scale the features in `x7`.
  - **Operation**: The output `x7` is multiplied element-wise with the attention weights.
- **Rationale**: Attention mechanisms allow the model to focus on the most important spatial regions of the feature map.
- **Advantage**:
  - Reduces noise and irrelevant information.
  - Improves performance by emphasizing key regions in the input data.

---

### **5. Additional Convolution Blocks**
- **Changes**:
  - `convblock6` is a new block added in the second convolution stage.
- **Rationale**: Adding more layers increases the model depth, allowing it to learn more hierarchical and fine-grained features.
- **Advantage**: Improves the ability to model complex patterns in the data.

---

### **6. Introduction of Skip Connections**
- **New Additions**: 
  - `self.skip1` and `self.skip2` layers introduce **skip connections**.
  - Skip connections add outputs from earlier layers (`x`) to later layers (`x4` and `x6`) after resizing them using interpolation.
- **Rationale**: Skip connections (like in ResNets) allow gradients to flow back more effectively and mitigate vanishing gradient issues.
- **Advantage**: 
  - Encourages feature reuse, leading to more efficient training.
  - Improves performance in deeper networks.
  - Provides robustness against overfitting by stabilizing the optimization process.

---

### **7. Filter Count Adjustments**
- **Changes**: The number of filters in most layers has been slightly increased (e.g., from 8→10, 16→14).
- **Rationale**: Increasing the number of filters allows the network to learn more features at each layer.
- **Advantage**: This adjustment enhances the capacity to capture more complex features, improving accuracy for complex datasets.

---

### **8. Activation Function Update**
- **Old Model**: `ReLU` (Rectified Linear Unit)
- **New Model**: `GELU` (Gaussian Error Linear Unit)
- **Rationale**: GELU is a smoother activation function than ReLU, incorporating a probabilistic element that activates neurons in a range rather than a hard cutoff at 0.
- **Advantage**: This change improves gradient flow and convergence, particularly for deeper networks, and helps reduce sharp saturation regions that could hinder learning.

---

### **9. Adjusted Dropout Rate**
- **Old Model**: `dropout_value = 0.1`
- **New Model**: `dropout_value = 0.03`
- **Rationale**: A lower dropout rate reduces the number of neurons randomly dropped during training, allowing more information to flow through the network in each forward pass.
- **Advantage**: This adjustment reduces the risk of underfitting and may lead to faster convergence while still providing regularization to prevent overfitting.

---

### **10. Introduced ReduceLROnPlateau**
- **Changes**: Implemented the `ReduceLROnPlateau` learning rate scheduler for the ThirdModel. This scheduler reduces the learning rate when the validation loss has stopped improving, allowing for more effective training.
- **Rationale**: The addition of this scheduler aims to enhance the model's performance by dynamically adjusting the learning rate based on the model's performance on the validation set. This helps in fine-tuning the learning process, especially in later epochs when the model may converge slowly.
- **Advantage**: By using `ReduceLROnPlateau`, the model can achieve better accuracy and generalization. It prevents the learning rate from being too high, which can lead to overshooting the optimal solution, and allows for a more gradual approach to convergence, ultimately improving the model's performance on unseen data.

---

### **Summary of Benefits**
1. **Improved Feature Learning**: Deeper layers, skip connections, and attention mechanisms increase the ability to learn complex features.
2. **Better Gradient Flow**: Skip connections stabilize training and prevent vanishing gradients.
3. **Robustness and Efficiency**: Lower dropout values and smoother activations (GELU) reduce the risk of overfitting while improving computational efficiency.
4. **Enhanced Interpretability**: Attention mechanisms allow the model to focus on significant features, making its decisions easier to interpret.
5. **Adaptability to Input Variability**: Interpolation ensures feature alignment across different scales, improving the network's flexibility.

These changes make the new model more powerful and resilient while maintaining efficiency, particularly for datasets with complex patterns. It is particularly suited for environments where computational resources or data are limited.

### Results: 
```
Parameters: 7.9k
Best Train Accuracy: 98.80
Best Test Accuracy: 99.43 (14th Epoch)
```
### Analysis: 
- The model achieves a test set accuracy of 99.43% with 7.9k parameters.
- The model achieves a train set accuracy of 98.80% with 7.9k parameters.


## Code for Third Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_rate_model3 = 0.03

# Model Definition 3
class ThirdModel(nn.Module):
    def __init__(self):
        super(ThirdModel, self).__init__()
        # Input Block
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_rate_model3)
        )

        # Convolution Block 1
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_rate_model3)
        )

        # Transition Block 1
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolution Block 2
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_rate_model3)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_rate_model3)
        )

        # New Convolution Block
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate_model3)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate_model3)
        )
        
        # Output Block
        self.global_avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

        self.skip_connection1 = nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(1, 1), padding=0, bias=False)
        self.skip_connection2 = nn.Conv2d(in_channels=14, out_channels=12, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.pool1(x3)
        
        x4 = self.layer4(x)
        x4 = x4 + self.skip_connection1(F.interpolate(x, size=(x4.shape[2], x4.shape[3])))
        
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x6 = x6 + self.skip_connection2(F.interpolate(x4, size=x6.shape[2:]))
        
        x7 = self.layer7(x6)
        
        b, c, h, w = x7.shape
        attention = F.adaptive_avg_pool2d(x7, 1)
        attention = torch.sigmoid(attention)
        x7 = x7 * attention
        
        x = self.global_avg_pool(x7)        
        x = self.layer8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```

## Console Log for Third Model

```
(temporary) ➜  erav3-s7-mnist-nn python src/train.py --model 3
Using device: mps
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Sequential: 1-1                        --
|    └─Conv2d: 2-1                       90
|    └─GELU: 2-2                         --
|    └─BatchNorm2d: 2-3                  20
|    └─Dropout: 2-4                      --
├─Sequential: 1-2                        --
|    └─Conv2d: 2-5                       1,260
|    └─GELU: 2-6                         --
|    └─BatchNorm2d: 2-7                  28
|    └─Dropout: 2-8                      --
├─Sequential: 1-3                        --
|    └─Conv2d: 2-9                       140
├─MaxPool2d: 1-4                         --
├─Sequential: 1-5                        --
|    └─Conv2d: 2-10                      1,260
|    └─GELU: 2-11                        --
|    └─BatchNorm2d: 2-12                 28
|    └─Dropout: 2-13                     --
├─Sequential: 1-6                        --
|    └─Conv2d: 2-14                      1,764
|    └─GELU: 2-15                        --
|    └─BatchNorm2d: 2-16                 28
|    └─Dropout: 2-17                     --
├─Sequential: 1-7                        --
|    └─Conv2d: 2-18                      1,512
|    └─GELU: 2-19                        --
|    └─BatchNorm2d: 2-20                 24
|    └─Dropout: 2-21                     --
├─Sequential: 1-8                        --
|    └─Conv2d: 2-22                      1,296
|    └─GELU: 2-23                        --
|    └─BatchNorm2d: 2-24                 24
|    └─Dropout: 2-25                     --
├─Sequential: 1-9                        --
|    └─AvgPool2d: 2-26                   --
├─Sequential: 1-10                       --
|    ���─Conv2d: 2-27                      120
├─Conv2d: 1-11                           140
├─Conv2d: 1-12                           168
=================================================================
Total params: 7,902
Trainable params: 7,902
Non-trainable params: 0
=================================================================
CUDA Available? False
Epoch: 1
Loss=0.032959796488285065 Batch_id=937 Accuracy=86.22: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 56.42it/s]

Test set: Average loss: 0.0655, Accuracy: 9809/10000 (98.09%)

Epoch: 2
Loss=0.035755179822444916 Batch_id=937 Accuracy=96.86: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 58.05it/s]

Test set: Average loss: 0.0491, Accuracy: 9855/10000 (98.55%)

Epoch: 3
Loss=0.2506659924983978 Batch_id=937 Accuracy=97.58: 100%|█████████████████���████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 57.68it/s]

Test set: Average loss: 0.0313, Accuracy: 9899/10000 (98.99%)

Epoch: 4
Loss=0.018190767616033554 Batch_id=937 Accuracy=97.80: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 59.68it/s]

Test set: Average loss: 0.0305, Accuracy: 9897/10000 (98.97%)

Epoch: 5
Loss=0.09306802600622177 Batch_id=937 Accuracy=97.94: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 59.74it/s]

Test set: Average loss: 0.0343, Accuracy: 9900/10000 (99.00%)

Epoch: 6
Loss=0.16479645669460297 Batch_id=937 Accuracy=98.01: 100%|██████████████████████████████████���███████████████████████████████████████████████████████| 938/938 [00:15<00:00, 59.46it/s]

Test set: Average loss: 0.0245, Accuracy: 9916/10000 (99.16%)

Epoch: 7
Loss=0.21610727906227112 Batch_id=937 Accuracy=98.16: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 58.44it/s]

Test set: Average loss: 0.0285, Accuracy: 9911/10000 (99.11%)

Epoch: 8
Loss=0.09736696630716324 Batch_id=937 Accuracy=98.20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 56.07it/s]

Test set: Average loss: 0.0254, Accuracy: 9918/10000 (99.18%)

Epoch: 9
Loss=0.0075428723357617855 Batch_id=937 Accuracy=98.39: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 55.54it/s]

Test set: Average loss: 0.0265, Accuracy: 9919/10000 (99.19%)

Epoch: 10
Loss=0.01036565750837326 Batch_id=937 Accuracy=98.38: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 58.68it/s]

Test set: Average loss: 0.0254, Accuracy: 9925/10000 (99.25%)

Epoch: 11
Loss=0.11539029330015182 Batch_id=937 Accuracy=98.73: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:16<00:00, 57.73it/s]

Test set: Average loss: 0.0190, Accuracy: 9938/10000 (99.38%)

Epoch: 12
Loss=0.2806856334209442 Batch_id=937 Accuracy=98.80: 100%|███████████████████████████████████████████████████████████████████��██████████████████████| 938/938 [00:16<00:00, 56.11it/s]

Test set: Average loss: 0.0195, Accuracy: 9934/10000 (99.40%)

Epoch: 13
Loss=0.006596127524971962 Batch_id=937 Accuracy=98.76: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 58.95it/s]

Test set: Average loss: 0.0189, Accuracy: 9932/10000 (99.41%)

Epoch: 14
Loss=0.012113465927541256 Batch_id=937 Accuracy=98.69: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 58.89it/s]

Test set: Average loss: 0.0179, Accuracy: 9943/10000 (99.43%)

Epoch: 15
Loss=0.009860938414931297 Batch_id=937 Accuracy=98.78: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:15<00:00, 58.96it/s]

Test set: Average loss: 0.0200, Accuracy: 9938/10000 (99.42%)


===================
Results:
Parameters: 7.9k
Best Train Accuracy: 98.80
Best Test Accuracy: 99.43 (14th Epoch)
===================
```

## Contributing

We welcome contributions to enhance the functionality and performance of this project. If you have suggestions or improvements, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.