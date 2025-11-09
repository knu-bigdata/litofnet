import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from train_model import train_LiTOFNet
from utils import make_trainloader
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # insert your pulse shape
    pulse_shape_one = np.loadtxt(r".\data\Middlebury_generation\pulse_example.csv", delimiter=",")

    pulse_len=len(pulse_shape_one)
    hist_len = 64

    batch=4096
    valid_batch=10000


    pulse_shape=torch.Tensor(pulse_shape_one)
    pulse_shape=pulse_shape.to(device)
    pulse_shape=pulse_shape.view([pulse_len])
    pulse_shape = pulse_shape.repeat(batch, 1, 1)  

    pulse_shape=pulse_shape.view([batch,pulse_len])

    valid_pulse_shape=torch.Tensor(pulse_shape_one)
    valid_pulse_shape=valid_pulse_shape.to(device)
    valid_pulse_shape=valid_pulse_shape.view([pulse_len])
    valid_pulse_shape = valid_pulse_shape.repeat(valid_batch, 1, 1)  

    valid_pulse_shape=valid_pulse_shape.view([valid_batch,pulse_len])

    # load train and valid data
    train_data_path = r".\data\train_sample\train_sample_data_frame_0000.parquet"
    valid_data_path = r".\data\valid_sample\valid_sample_data_frame_0000.parquet"
    trainloader = make_trainloader(train_data_path, batch)
    validloader = make_trainloader(valid_data_path, valid_batch)

    # model definition
    model = train_LiTOFNet(filter1 = 14, 
                           filter2 = 42, 
                           kernel1 = 12, 
                           kernel2 = 3,
                           conv1 = 3, 
                           conv2 = 2, 
                           num_nureon = 19, 
                           pulse_len = pulse_len , 
                           hist_len = 64).to(device)
    
    #train configuration
    save_model_state_path = "./check_point"
    os.makedirs(save_model_state_path , exist_ok=True)
    lr = 0.00311
    loss_func= nn.MSELoss().to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 500
    patience_limit = 100
    patience = 0
    best_loss=0
    iter=0
    loss_list=[]
    epoch_loss=[]

    for epoch in range(num_epochs) :
        if(patience==patience_limit):
            break
        for i, (hist, ftof, _ ) in enumerate(trainloader) : 
            optimizer.zero_grad()
            hist= hist.to(device)
            ftof=ftof.view([ftof.size(0)])
            ftof=ftof.to(device)
            outputs = model(pulse_shape, hist)    
            loss_siam=loss_func(outputs, ftof)
            
            loss_list.append(loss_siam.item())  

            # print(outputs, labels)
            loss_siam.backward()
            optimizer.step()

            
            iter += 1
                        
        epoch_loss.append(loss_siam.item())


        with torch.no_grad():
            model.eval()
            valid_r2score=[]
            loss_valid=0
            valid_iter = 0
            print(f'epoch : {epoch}. model_Loss : ', epoch_loss[epoch])
            for valid_hist, valid_ftof , _ in validloader : 
                valid_hist=valid_hist.to(device)
                valid_ftof=valid_ftof.view([valid_ftof.size(0)])
                valid_ftof= valid_ftof.to(device)

                outputs_valid = model(valid_pulse_shape, valid_hist)   

                temp_loss=loss_func(outputs_valid , valid_ftof)
                loss_valid+=temp_loss.item()   
            
                valid_r2score.append(r2_score(valid_ftof.detach().cpu().numpy(), outputs_valid.detach().cpu().numpy()))  # FTOF 값과 interpolatin한 값과 비교        
                valid_iter +=1       

            if(epoch==0):
                best_loss=loss_valid
            else :
                if(best_loss< loss_valid):
                    patience+=1
                else :
                    best_loss=loss_valid
                    patience=0
                    torch.save(model.state_dict(), save_model_state_path+f'/model_{epoch}_point.pth')    
            
            print("Mean of R2score: ", torch.tensor(valid_r2score).mean() ,"valid_loss: ", loss_valid/valid_iter, "best_loss", best_loss/valid_iter ,"patience", patience)

if __name__ == "__main__":
    main()    



