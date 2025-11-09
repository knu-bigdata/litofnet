import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation_model import eval_LiToFNet
from utils import make_dataloader
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    # Set the device to GPU if available, otherwise use CPU.
    output_data_folder = './output/'
    os.makedirs(output_data_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CTOF_RES=1000/75


    # Load the dataset
    data = pd.read_csv('./data/fine_histogram.csv' , index_col=0 )
    sub_network1_output = pd.read_csv('./data/subnetwork1_output.csv', header=None)
    sub_network1_output=torch.tensor(sub_network1_output.to_numpy().reshape(19)).to(device).to(torch.bfloat16)

    #Load the LiToFNet model with the specified parameters.        
    LiToFNet_model = eval_LiToFNet(14,42,12,3,3,2,126,19)
    LiToFNet_model.load_state_dict(torch.load('./weights/litofnet_weights.pth', weights_only=True))

    # Load the output of sub-network 1 into the LiToFNet model.
    with torch.no_grad():
        LiToFNet_model.fc1.bias = torch.nn.Parameter(sub_network1_output)

    LiToFNet_model.to(device).to(torch.bfloat16)

    # Create a DataLoader for the dataset.
    batch_size = 11200 # 56 by 200 piexel per 1 frame

    example_loader = make_dataloader(data, batch_size, device=device) 

    

    with torch.no_grad():
        LiToFNet_model.eval()
        for i,(fine_hist, CTOF_VALUE) in enumerate(example_loader) :
            fine_hist = fine_hist.to(device).to(torch.bfloat16)
            FTOF_VALUE=LiToFNet_model(fine_hist)
            TOF_VALUE=FTOF_VALUE.view(batch_size,1)+CTOF_VALUE.view(batch_size,1)*CTOF_RES
            TOF_VALUE[CTOF_VALUE==0]=0
            if i == 0:
                TOF_DATA = pd.DataFrame(data=TOF_VALUE.cpu().numpy().reshape(1,batch_size))
            else :
                TOF_DATA = pd.concat([TOF_DATA, pd.DataFrame(TOF_VALUE.cpu().numpy().reshape(1, batch_size))], ignore_index=True)

    #visualize the output TOF_DATA
    raw_data=TOF_DATA.to_numpy().reshape(56,200)[2:54 ,4:196] 
    rotated_image = np.rot90(raw_data , 2)
    #Delete dummy data and create a 2D depth image
    norm = mcolors.Normalize(vmin=0, vmax=70)

    cmap = cm.jet

    colored_data = cmap(norm(rotated_image*0.15))

    fig, ax = plt.subplots()
    im = ax.imshow(colored_data)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Distance (m)', fontsize=12)

    ax.set_title("2D Depth image")
    ax.axis('off')

    plt.imshow(colored_data)

    plt.savefig(output_data_folder+"/2d_depth_image.png", format="png", bbox_inches='tight')  
    np.savetxt(output_data_folder+"/tof_data.csv" ,rotated_image, delimiter=',', fmt='%.4f')          

if __name__ == "__main__":
    main()            
            


