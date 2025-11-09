import torch
import torch.nn as nn
import torch.nn.functional as F
from subnetwork import Subnetwork

class eval_LiToFNet(nn.Module):
    
 # A network based on a Siamese architecture for Time-of-Flight (ToF) depth calculation.

    def __init__(self,filter1, filter2, kernel1, kernel2,conv1, conv2, feature_map_size, num_neuron):
        # Initialization function for the LiToFNet model.
        super(eval_LiToFNet, self).__init__()
        self.subnet = Subnetwork(filter1,filter2,kernel1, kernel2,conv1, conv2)
        self.fc1 = nn.Linear(feature_map_size,num_neuron)
        self.fc2 = nn.Linear(num_neuron, 1)
      
        
    def _initialize_weights(self):
        # Weight initialization function 
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
 
              
    def forward(self, y_input):
        # The forward function for the model.
        y_feature_maps = self.subnet(y_input)
        feature_maps_diff = y_feature_maps
        x = F.relu(self.fc1(feature_maps_diff))
        x = self.fc2(x)
        return x.squeeze(-1)
    
