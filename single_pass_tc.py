import os
import torch
import numpy as np
from tqdm import tqdm
import cv2

from TransformerModel.model import TransformerNet
import trainingdata.utils as utils
import video_utils
from config import get_video_config

def single_pass_stylize(video_config, filename: str, debug_dir: str = "debug_output", batch_size: int = 4):
    """
    Single pass video stylization with temporal consistency and debugging outputs
    """
    # Extract video and model names for unique outputs
    video_name = os.path.splitext(filename)[0]
    model_name = os.path.splitext(os.path.basename(video_config['model_name']))[0]
    unique_prefix = f"{video_name}_{model_name}"
    
    # Create unique debug directory
    debug_dir = f"{debug_dir}_{unique_prefix}"
    os.makedirs(debug_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerNet().to(device)
    state_dict = torch.load(os.path.join(
        video_config['model_binaries_path'], 
        video_config['model_name']
    ))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    print("Processing video frames...")
    cap = cv2.VideoCapture(video_config['input_path']) 
    fps = 24  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_config['input_path']}")
    
    frames_to_process = min(int(5 * fps), total_frames)  # 5 seconds worth of frames
    
    stylized_frames = []
    prev_stylized = None
    prev_frame = None
    
    # Process frames in batches
    for batch_start in tqdm(range(0, frames_to_process, batch_size)):
        if batch_start % (batch_size * 5) == 0: 
            torch.cuda.empty_cache()
            
        batch_frames = []
        batch_frames_raw = []
        for i in range(batch_size):
            frame_idx = batch_start + i
            if frame_idx >= frames_to_process:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (video_config['img_width'], video_config['img_width']))
            batch_frames_raw.append(frame)
            
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            batch_frames.append(frame_tensor)
        
        if not batch_frames:
            break
            
        # Move batch to GPU
        batch_tensor = torch.stack(batch_frames).to(device)
        
        with torch.no_grad():
            batch_stylized = model(batch_tensor)
            
            # Process each frame in batch
            for i, current_stylized in enumerate(batch_stylized):
                frame_idx = batch_start + i
                current_frame = batch_frames_raw[i]
                
                if prev_stylized is not None and prev_frame is not None:
                    # Compute optical flow
                    flow_forward = video_utils.estimate_optical_flow(
                        prev_frame,
                        current_frame,
                        method='dualtvl1'
                    ).to(device)
                    
                    flow_backward = video_utils.estimate_optical_flow(
                        current_frame,
                        prev_frame,
                        method='dualtvl1'
                    ).to(device)
                    
                    # Apply smoothing to flow
                    flow_forward_np = flow_forward.cpu().numpy()
                    for j in range(2):
                        flow_forward_np[j] = cv2.GaussianBlur(
                            flow_forward_np[j],
                            ksize=(5, 5),
                            sigmaX=1.5
                        )
                    flow_forward = torch.from_numpy(flow_forward_np).to(device)
                    
                    # Save flow visualization
                    if frame_idx % 10 == 0:
                        flow_magnitude = np.sqrt(np.sum(flow_forward_np**2, axis=0))
                        utils.save_image(
                            normalize_flow(flow_magnitude, clip_percentile=2),
                            os.path.join(debug_dir, f"frame_{frame_idx:04d}_flow.png")
                        )
                    
                    # Compute consistency mask
                    mask = video_utils.compute_consistency_mask(
                        flow_forward,
                        flow_backward,
                        threshold=0.08
                    ).to(device)
                    
                    # Apply additional smoothing to mask
                    mask_np = mask.cpu().numpy()
                    mask_np = cv2.GaussianBlur(mask_np, (7, 7), 2.0)
                    mask = torch.from_numpy(mask_np).to(device)
                    
                    if frame_idx % 10 == 0:
                        utils.save_image(
                            mask_np,
                            os.path.join(debug_dir, f"frame_{frame_idx:04d}_mask.png")
                        )
                    
                    # Warp previous stylized frame
                    prev_stylized = torch.clamp(prev_stylized, 0.0, 1.0)
                    warped_prev = video_utils.warp_image(
                        prev_stylized,
                        flow_forward
                    ).to(device)
                    warped_prev = torch.clamp(warped_prev, 0.0, 1.0)
                    
                    if frame_idx % 10 == 0:
                        print(f"prev_stylized min/max: {prev_stylized.min():.4f}/{prev_stylized.max():.4f}")
                        print(f"flow_forward min/max: {flow_forward.min():.4f}/{flow_forward.max():.4f}")
                        print(f"warped_prev min/max: {warped_prev.min():.4f}/{warped_prev.max():.4f}")
                        
                        utils.save_image(
                            warped_prev,
                            os.path.join(debug_dir, f"frame_{frame_idx:04d}_warped.png")
                        )
                    
                    # Blend current stylized frame with warped previous frame
                    alpha = 0.8
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    current_stylized = (
                        alpha * current_stylized + 
                        (1 - alpha) * warped_prev * mask + 
                        current_stylized * (1 - mask)
                    )
                    
                    current_stylized = torch.clamp(current_stylized, 0.0, 1.0)
                
                # Save debug outputs
                if frame_idx % 10 == 0:
                    utils.save_image(
                        current_stylized,
                        os.path.join(debug_dir, f"frame_{frame_idx:04d}_final.png")
                    )
                
                # Update previous frame references
                prev_frame = current_frame
                prev_stylized = current_stylized
                # Convert to correct format before appending
                frame_to_append = (current_stylized.squeeze(0).cpu().detach() * 255).permute(1, 2, 0).numpy().astype(np.uint8)
                stylized_frames.append(frame_to_append)
    
    cap.release()
    
    # Save video with unique name
    output_filename = f"stylized_single_pass_{unique_prefix}.mp4"
    output_path = os.path.join(video_config['output_path'], output_filename)
    utils.save_video(stylized_frames, output_path, fps=24)
    
    return stylized_frames

def normalize_flow(flow, clip_percentile=2):
    """
    Normalize flow values to [0, 1] range with outlier clipping
    """
    # Clip outliers
    low = np.percentile(flow, clip_percentile)
    high = np.percentile(flow, 100 - clip_percentile)
    flow = np.clip(flow, low, high)
    
    # Normalize to [0, 1]
    flow_min = flow.min()
    flow_max = flow.max()
    if flow_max - flow_min > 0:
        return (flow - flow_min) / (flow_max - flow_min)
    return flow

if __name__ == "__main__":
    video_config = get_video_config()
    os.makedirs(video_config['output_path'], exist_ok=True)
    single_pass_stylize(video_config, "test-video-1.mp4", debug_dir="debug_output")
