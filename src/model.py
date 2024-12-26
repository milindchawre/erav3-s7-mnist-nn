import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define different dropout rates for each model
dropout_rate_model1 = 0.03
dropout_rate_model2 = 0.1
dropout_rate_model3 = 0.03

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

