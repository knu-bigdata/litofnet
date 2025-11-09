
import torch
import torch.nn as nn
import torch.nn.functional as F


class Subnetwork(nn.Module):
# The Subnetwork class implements a simple 1D convolutional network followed by a fully connected layer.
    
    def __init__(self,filter1, filter2, kernel1, kernel2,conv1, conv2):
        # Initializes the Subnetwork with the given parameters.
        super(Subnetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, filter1, kernel_size=kernel1, stride=conv1, padding=0)
        self.conv2 = nn.Conv1d(filter1, filter2, kernel_size=kernel2, stride=conv2, padding=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)    
        self._initialize_weights()

    def _initialize_weights(self):
        #Initializes the weights of convolutional layers 
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Defines the forward pass of the network, applying convolutions, pooling, and fully connected layers.
        x = x.unsqueeze(1)  
        
        x  = self.conv1(x)  #Convolutional layer 1
        
        x = F.relu(x)       # Activation function
        
        x = self.avgpool(x) # Average pooling 
        
        x  = self.conv2(x)  #Convolutional layer 2
        
        x = F.relu(x)       # Activation function
        
        x = self.avgpool(x) # Average pooling
        
        feature_maps = x.flatten(1)  # Flatten the feature maps for the fully connected layer
        
        
        return feature_maps
