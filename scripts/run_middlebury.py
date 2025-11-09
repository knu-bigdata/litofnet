import os
import gc
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation_model import eval_LiToFNet
from utils import save_depth_images
import numpy as np
import pandas as pd
import torch
from time import perf_counter as tpc
from tqdm import tqdm
import pyarrow.parquet as pq
import concurrent.futures as cf

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 1. Compute bounding box from the first frame =====
def compute_bbox_from_frame0(datafolder, scene_name, spp, bpp):
    path = f"{datafolder}\\{scene_name}/{spp}_{bpp}/sample_data_{scene_name}_{spp}_{bpp}_frame_0000.parquet"
    tbl = pq.read_table(
        path,
        columns=["bins", "v_idx", "h_idx" ],
        use_threads=True, memory_map=True, pre_buffer=True
    )
    bins_arr = tbl["bins"].combine_chunks()
    BIN = bins_arr.type.list_size
    n = tbl.num_rows
    bins = np.asarray(bins_arr.values, dtype=np.uint16).reshape(n, BIN)

    v = tbl["v_idx"].to_numpy()
    h = tbl["h_idx"].to_numpy()

    nz = (bins.sum(axis=1) != 0)
    if not np.any(nz):
        raise RuntimeError("All histograms contain only zeros. Please check your data or file path.")
    v_nz = v[nz]; h_nz = h[nz]
    y_min, y_max = int(v_nz.min()), int(v_nz.max()) + 1
    x_min, x_max = int(h_nz.min()), int(h_nz.max()) + 1

    new_nr = y_max - y_min
    new_nc = x_max - x_min
    return (y_min, y_max, x_min, x_max, new_nr, new_nc)

# 2. Use bounding box to quickly read only the target frame data =====
def load_frame_cropped_arrays(datafolder, scene_name, spp, bpp, frame, bbox):
    y_min, y_max, x_min, x_max, new_nr, new_nc = bbox
    path = f"{datafolder}\\{scene_name}/{spp}_{bpp}/sample_data_{scene_name}_{spp}_{bpp}_frame_{frame:04d}.parquet"

    # Parquet row-level filtering (pushdown)
    filters = [
        ("v_idx", ">=", y_min), ("v_idx", "<", y_max),
        ("h_idx", ">=", x_min), ("h_idx", "<", x_max)
    ]
    tbl = pq.read_table(
        path,
        columns=["bins", "v_idx", "h_idx", "DISTANCE" ,"CTOF"],  # Only load necessary columns
        filters=filters,
        use_threads=True, memory_map=True, pre_buffer=True
    )

    bins_arr = tbl["bins"].combine_chunks()
    BIN = bins_arr.type.list_size
    n = tbl.num_rows
    expected = new_nr * new_nc  # Expected number of rows

    if n != expected:
        # (Usually sorted if preprocessed from Middlebury dataset)
        # Uncomment below if sorting is required:
        # v = tbl["v_idx"].to_numpy(); h = tbl["h_idx"].to_numpy()
        # order = np.lexsort((h, v))
        # bins_vals = np.asarray(bins_arr.values, dtype=np.uint16).reshape(n, BIN)[order]
        # dst = tbl["DISTANCE"].to_numpy()[order]
        # return bins_vals.reshape(new_nr*new_nc, BIN), dst.astype(np.float32, copy=False)
        pass

    bins_vals = np.asarray(bins_arr.values, dtype=np.uint16)
    # Ensure writable, C-contiguous array (copy if read-only or non-contiguous)
    if (not bins_vals.flags.writeable) or (not bins_vals.flags.c_contiguous):
        bins_vals = np.ascontiguousarray(bins_vals)
    bins_vals = bins_vals.reshape(n, BIN)
    ctof = tbl["CTOF"].to_numpy().astype(np.float32, copy=False)
    dst = tbl["DISTANCE"].to_numpy()  # (N,)
    return bins_vals, dst.astype(np.float32, copy=False), ctof, new_nr, new_nc  # Flattened output

# ===== 3) Process a single frame using vectorized inference =====
def run_one_frame_vectorized(
    test_litof, device,
    hist_flat_np, dst_flat_np, ctof_flat_np, new_nr, new_nc,
    depth_mm, real_mm, frame_idx,
    AMP_DTYPE=torch.float16, CHUNK=1048576
):
    """
    hist_flat_np: (N, 64) uint16
    dst_flat_np : (N,)   float32
    ctof_flat_np: (N,)   float32
    """
     
    N = hist_flat_np.shape[0]
    # Fixed CPU buffer (pinned memory) to receive GPU→CPU results
    out_depth_cpu = torch.empty(N, dtype=torch.float32, device="cpu", pin_memory=True)

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=(device.type=='cuda')):
        off = 0
        while off < N:
            i0, i1 = off, min(off + CHUNK, N)

            # 1) Ensure contiguous array slice
            view = hist_flat_np[i0:i1]
            if (not view.flags.writeable) or (not view.flags.c_contiguous):
                view = np.ascontiguousarray(view)

            # 2) CPU → GPU copy (H2D transfer)
            hist_chunk = torch.as_tensor(view, device=device, dtype=AMP_DTYPE)

            # 3) Inference
            depth_est = test_litof(hist_chunk)   # (B,)

            # === Add CTOF offset ===
            ctof_chunk = torch.as_tensor(ctof_flat_np[i0:i1], device=device, dtype=torch.float32)
            depth_est = depth_est.to(torch.float32) + ctof_chunk

            # 4) GPU → pinned memory (direct copy, no intermediate CPU tensor)
            if device.type == 'cuda':
                out_depth_cpu[i0:i1].copy_(depth_est.to(dtype=torch.float32), non_blocking=True)
            else:
                out_depth_cpu[i0:i1].copy_(depth_est.to(dtype=torch.float32))

            del hist_chunk, depth_est
            off = i1

    # Write results as 2D arrays (fast contiguous writes)
    depth_mm[frame_idx] = out_depth_cpu.numpy().reshape(new_nr, new_nc)
    real_mm[frame_idx]  = dst_flat_np.reshape(new_nr, new_nc)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_dim = 64
    batch = 1000
    sub_network1_output = pd.read_csv('./data/subnetwork1_output.csv', header=None)
    sub_network1_output=torch.tensor(sub_network1_output.to_numpy().reshape(19)).to(device).to(torch.bfloat16)

    #Load the LiToFNet model with the specified parameters.        
    LiToFNet_model = eval_LiToFNet(14,42,12,3,3,2,126,19)
    LiToFNet_model.load_state_dict(torch.load('./weights/litofnet_weights.pth', weights_only=True))

    # Load the output of sub-network 1 into the LiToFNet model.
    with torch.no_grad():
        LiToFNet_model.fc1.bias = torch.nn.Parameter(sub_network1_output)

    LiToFNet_model.to(device).to(torch.bfloat16)
    LiToFNet_model.eval()
    #scenes = ['Reindeer', 'Art', 'Plastic', 'Moebius', 'Laundry', 'Dolls', 'Bowling1', 'Books']
    scenes = ['Art']
    simulation_params = np.array([
        [1000, 50]
    ])

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    AMP_DTYPE = torch.bfloat16
    
    # Load middelbury data and save image
    datafolder = "./data/Middlebury/histogram"
    OUT_DIR = "./output/Middlebury_benchmark"
    save_image_folder = "./output/Middlebury_benchmark/image"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(save_image_folder, exist_ok=True)

    FRAMES = 1

    for scene_name in scenes:
        for spp, bpp in simulation_params:
            # (A) Compute bbox from the first frame
            bbox = compute_bbox_from_frame0(datafolder, scene_name, spp, bpp)
            y_min, y_max, x_min, x_max, new_nr, new_nc = bbox
            print(f"[{scene_name} {spp}/{bpp}] bbox = y[{y_min}:{y_max}] x[{x_min}:{x_max}] -> {new_nr}x{new_nc}")

            # (B) Create memmap files for (scene, spp, bpp)
            depth_path = os.path.join(OUT_DIR, f"{scene_name}_{spp}_{bpp}_depth_img.npy")
            real_path  = os.path.join(OUT_DIR, f"{scene_name}_{spp}_{bpp}_real_depth_img.npy")

            mode = "r+" if os.path.exists(depth_path) else "w+"

            depth_mm = np.lib.format.open_memmap(
                depth_path, mode=mode, dtype=np.float32, shape=(FRAMES, new_nr, new_nc)
            )
            real_mm  = np.lib.format.open_memmap(
                real_path,  mode=mode, dtype=np.float32, shape=(FRAMES, new_nr, new_nc)
            )

            # (C) Overlap I/O and computation (prefetch next frame)
            SHOW_TIMING = True

            with cf.ThreadPoolExecutor(max_workers=1) as ex, \
                tqdm(total=FRAMES, desc=f"{scene_name} {spp}/{bpp}", unit="frame", leave=True) as pbar:

                # Load frame 0
                hist0, dst0, ctof0, _, _ = load_frame_cropped_arrays(datafolder, scene_name, spp, bpp, 0, bbox)
                next_future = ex.submit(load_frame_cropped_arrays, datafolder, scene_name, spp, bpp, 1, bbox)

                total_time = 0.0

                # Run inference for frame 0
                t0 = tpc()
                run_one_frame_vectorized(
                    LiToFNet_model, device,
                    hist0, dst0,ctof0, new_nr, new_nc,
                    depth_mm, real_mm, frame_idx=0,
                    AMP_DTYPE=AMP_DTYPE, CHUNK=65536
                )
                dt = tpc() - t0
                total_time += dt
                pbar.update(1)
                if SHOW_TIMING:
                    pbar.set_postfix(last_s=f"{dt:.3f}", avg_s=f"{total_time/1:.3f}", fps=f"{1/(total_time/1):.1f}")
                print(f"✅ Saved: {depth_path} [frame=0000] ({dt:.3f}s)")

                # Remaining frames
                for frame in range(1, FRAMES):
                    t0 = tpc()

                    hist, dst,  ctof, _, _ = next_future.result()
                    if frame + 1 < FRAMES:
                        next_future = ex.submit(load_frame_cropped_arrays, datafolder, scene_name, spp, bpp, frame+1, bbox)

                    run_one_frame_vectorized(
                        LiToFNet_model,  device,
                        hist, dst,ctof, new_nr, new_nc,
                        depth_mm, real_mm, frame_idx=frame,
                        AMP_DTYPE=AMP_DTYPE, CHUNK=65536
                    )

                    # Periodic flush and garbage collection
                    if frame % 20 == 0:
                        depth_mm.flush(); real_mm.flush()
                    if frame % 50 == 0:
                        gc.collect()

                    dt = tpc() - t0
                    total_time += dt
                    pbar.update(1)
                    if SHOW_TIMING:
                        avg = total_time / (frame + 1)
                        pbar.set_postfix(last_s=f"{dt:.3f}", avg_s=f"{avg:.3f}", fps=f"{1/avg:.1f}")

                    if frame % 100 == 0:
                        print(f"✅ Saved: {depth_path} [frame={frame:04d}] ({dt:.3f}s)")

            # Final cleanup
            depth_mm.flush(); real_mm.flush()
            del depth_mm, real_mm
            gc.collect()

            real_path  = os.path.join(OUT_DIR, f"{scene_name}_{spp}_{bpp}_real_depth_img.npy")

        save_depth_images(depth_path , real_path, save_image_folder , f"{scene_name}_{spp}_{bpp}")    








if __name__ == '__main__':
    main()
