import torch.nn as nn
import torch.nn.functional as F

class train_SubNetwork(nn.Module):
    def __init__(self,filter1, filter2, kernel1, kernel2,conv1, conv2 , pulse_len, hist_len):
        super(train_SubNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, filter1, kernel_size=kernel1, stride=conv1, padding=0)
        self.conv2 = nn.Conv1d(filter1, filter2, kernel_size=kernel2, stride=conv2, padding=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.pulse_len = pulse_len
        self.fc1 = nn.Linear(pulse_len,hist_len)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x ):
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length)
        if(x.size()==(x.shape[0],1,self.pulse_len)):
             x=self.fc1(x)

        x = self.avgpool(F.relu(self.conv1(x)))  # Output shape: (batch_size, filter1, L1)

        x = self.avgpool(F.relu(self.conv2(x)))  # Output shape: (batch_size, filter2, L2)
        
        feature_maps = x.flatten(1)  # Flattened feature maps: (batch_size, filter2 * L2)
        return feature_maps
    
class train_LiTOFNet(nn.Module):
    def __init__(self,filter1, filter2, kernel1, kernel2,conv1, conv2, num_nureon, pulse_len , hist_len):
        super(train_LiTOFNet, self).__init__()
        conv1_feature = int((int((hist_len - kernel1)/conv1) + 1)/2)
        self.feature_map_size= int(((int((conv1_feature - kernel2 + 2*2)/conv2) + 1)/2) * filter2)

        self.subnet = train_SubNetwork(filter1,filter2,kernel1, kernel2,conv1, conv2, pulse_len, hist_len)

        self.fc1 = nn.Linear(self.feature_map_size,num_nureon)
        self.fc2 = nn.Linear(num_nureon, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, F_input, y_input ):
        F_feature_maps = self.subnet(F_input)
        y_feature_maps = self.subnet(y_input)

        # Compute differences
        feature_maps_diff = y_feature_maps-F_feature_maps # (batch_size, feature_map_size)

        # Pass through fully connected layers

        x = F.relu(self.fc1(feature_maps_diff))
        x = self.fc2(x)

        return x.squeeze(-1)
