import os
import torch
import torch.nn as nn
import pandas as pd
from numba import njit
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



def make_dataloader(dataset, batch_size, device ):  # 56 by 200 piexel per 1 frame
    fine_hist = dataset.iloc[:, :64].to_numpy() # Fine histogram data
    CTOF_VALUE = dataset.loc[:, 'CTOF'].to_numpy() # CTOF value measured by the LiDAR sensor.

    fine_hist_tnesor =torch.FloatTensor(fine_hist).to(device)
    CTOF_VALUE_tensor = torch.FloatTensor(CTOF_VALUE).to(device)
    dataset = TensorDataset(fine_hist_tnesor, CTOF_VALUE_tensor)

    dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=True)

    return dataloader

@njit
def refined_end_gt_with_validation(hist_array, start_array, zero_tolerance=3,
                                   region_width=16, region_step=8):
    B, L = hist_array.shape
    end_array = np.zeros(B, dtype=np.int32)
    is_valid_mask = np.ones(B, dtype=np.bool_)  # validity mask (True = valid)

    for b in range(B):
        h = hist_array[b]
        s = int(start_array[b])
        if s < 0 or s >= L:
            is_valid_mask[b] = False
            end_array[b] = L - 1
            continue

        # 1. Compute adaptive threshold ===
        min_region_mean = 1e9
        num_regions = (L - region_width) // region_step + 1
        for r in range(num_regions):
            start_idx = r * region_step
            end_idx = start_idx + region_width
            if end_idx > L:
                break
            region_sum = 0.0
            for i in range(start_idx, end_idx):
                region_sum += h[i]
            region_mean = region_sum / region_width
            if region_mean < min_region_mean:
                min_region_mean = region_mean
        threshold = min_region_mean

        # 2. Find peak position 
        peak = s
        max_val = h[s]
        for i in range(s + 1, L):
            if h[i] > max_val:
                max_val = h[i]
                peak = i

        # 3. Estimate end position 
        zero_count = 0
        end_gt = L - 1  # fallback
        for i in range(peak + 1, L):
            curr = h[i]
            if curr == 0:
                zero_count += 1
                if zero_count >= zero_tolerance:
                    end_gt = i
                    break
            elif curr > threshold:
                zero_count = 0
            elif curr < threshold / 2:
                if zero_count <= zero_tolerance:
                    end_gt = i
                    break
        end_array[b] = end_gt

        # 4. Validation conditions 
        local_max = np.max(h[s:min(end_gt + 1, L)])
        global_max = np.max(h)
        if local_max < global_max:
            is_valid_mask[b] = False

        peak_count = 0
        tol = 1e-6
        for i in range(L):
            if abs(h[i] - max_val) < tol:
                peak_count += 1
        if peak_count >= 3:
            is_valid_mask[b] = False 
            

    return is_valid_mask



def make_trainloader(data_path, batchsize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    full_df = pd.read_parquet(data_path)


    # Extract input (bins) and target (FTOF)
    # Ensure all bins have the same length
    lengths = full_df['bins'].apply(len)
    most_common_len = lengths.mode()[0]
    filtered_df = full_df[lengths == most_common_len]

    # Convert bins column into 2D numpy array
    bins_array = np.stack(filtered_df['bins'].values).astype(np.float32)

    #Delete failed histograms
    train_mask=refined_end_gt_with_validation(bins_array , np.array(filtered_df['FTOF']))
    filtered_df = filtered_df[train_mask].reset_index(drop=True)
    bins_array = np.stack(filtered_df['bins'].values).astype(np.float32)


    train_hist = np.array(bins_array)


    train_CTOF = np.array(filtered_df['CTOF']).reshape(-1,1)



    train_FTOF = np.array(filtered_df['FTOF']).reshape(-1,1)


    merge_testX_tensor=torch.FloatTensor(train_hist)
    merge_testX_tensor = merge_testX_tensor.to(device)
    merge_testY_tensor = torch.FloatTensor( train_FTOF)
    merge_testY_tensor = merge_testY_tensor.to(device)
    merge_testZ_tensor = torch.Tensor(train_CTOF)
    merge_testZ_tensor = merge_testZ_tensor.to(device)


    merge_train_dataset= TensorDataset(merge_testX_tensor, merge_testY_tensor, merge_testZ_tensor)
    train_loader = DataLoader(merge_train_dataset, batch_size=batchsize, shuffle=True, num_workers=0 , drop_last=True)

    return train_loader




def save_depth_images(OUT_DIR, real_DIR, SAVE_DIR , name):
    """
    Loads predicted and real depth .npy files and saves comparison images.

    Parameters:
        OUT_DIR (str): Path to predicted depth .npy file.
        real_DIR (str): Path to real depth .npy file.
        SAVE_DIR (str): Output directory to save resulting images.
    """

    # Create output folder if not exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load memory-mapped arrays
    depth_mm = np.load(OUT_DIR, mmap_mode="r")
    real_mm  = np.load(real_DIR,  mmap_mode="r")

    print("depth shape:", depth_mm.shape)
    print("real  shape:", real_mm.shape)

    FRAMES, H, W = depth_mm.shape

    for frame in range(FRAMES):
        zero = real_mm == 0
        depth_mm_copy = depth_mm.copy()
        depth_mm_copy[zero] = 0

        a = depth_mm_copy[frame] * 0.15
        b = real_mm[frame]

        vmin = np.nanmin(b)
        vmax = np.nanmax(b)


        fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

        im1 = axs[0].imshow(a, vmin=vmin, vmax=vmax, cmap=plt.cm.viridis)
        axs[0].set_title(f"Ours")
        axs[0].axis("off")

        im2 = axs[1].imshow(b, vmin=vmin, vmax=vmax, cmap = plt.cm.viridis)
        axs[1].set_title(f"Ground Truth")
        axs[1].axis("off")

        save_path = os.path.join(SAVE_DIR, f"{name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Saved: {save_path}")











