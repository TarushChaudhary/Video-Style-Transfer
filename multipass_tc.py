import os
import torch
import numpy as np
from tqdm import tqdm
import cv2

from TransformerModel.model import TransformerNet
import trainingdata.utils as utils
import video_utils
from config import get_video_config

def multi_pass_stylize(video_config, filename: str, num_passes: int = 3, debug_dir: str = "debug_output", batch_size: int = 4):
    """
    Multi-pass video stylization with temporal consistency and debugging outputs
    """
    video_name = os.path.splitext(filename)[0]
    model_name = os.path.splitext(os.path.basename(video_config['model_name']))[0]
    unique_prefix = f"{video_name}_{model_name}"
    
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

    print("Reading video frames...")
    cap = cv2.VideoCapture(video_config['input_path'])
    fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_config['input_path']}")
        
    frames_to_process = min(int(8 * fps), total_frames)
    
    raw_frames = []
    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (video_config['img_width'], video_config['img_width']))
        raw_frames.append(frame)
    cap.release()

    stylized_frames = None
    
    # Multiple passes
    for pass_idx in range(num_passes):
        print(f"Starting pass {pass_idx + 1}/{num_passes}")
        current_pass_frames = []
        prev_stylized = None
        prev_frame = None
        
        for batch_start in tqdm(range(0, len(raw_frames), batch_size)):
            if batch_start % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
            
            batch_frames = []
            batch_frames_raw = raw_frames[batch_start:batch_start + batch_size]
            
            for frame in batch_frames_raw:
                frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
                batch_frames.append(frame_tensor)
            
            batch_tensor = torch.stack(batch_frames).to(device)
            
            with torch.no_grad():
                batch_stylized = model(batch_tensor)
                
                # Process each frame in batch
                for i, current_stylized in enumerate(batch_stylized):
                    frame_idx = batch_start + i
                    current_frame = batch_frames_raw[i]
                    
                    if prev_stylized is not None and prev_frame is not None:
                        # Compute bidirectional optical flow
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
                        
                        # Smooth flow
                        flow_forward_np = flow_forward.cpu().numpy()
                        for j in range(2):
                            flow_forward_np[j] = cv2.GaussianBlur(
                                flow_forward_np[j],
                                ksize=(5, 5),
                                sigmaX=1.5
                            )
                        flow_forward = torch.from_numpy(flow_forward_np).to(device)
                        
                        # Save flow visualization for last pass
                        if pass_idx == num_passes - 1 and frame_idx % 10 == 0:
                            flow_magnitude = np.sqrt(np.sum(flow_forward_np**2, axis=0))
                            utils.save_image(
                                normalize_flow(flow_magnitude, clip_percentile=2),
                                os.path.join(debug_dir, f"frame_{frame_idx:04d}_flow_pass{pass_idx+1}.png")
                            )
                        
                        # Compute and smooth consistency mask
                        mask = video_utils.compute_consistency_mask(
                            flow_forward,
                            flow_backward,
                            threshold=0.1
                        ).to(device)
                        
                        mask_np = mask.cpu().numpy()
                        mask_np = cv2.GaussianBlur(mask_np, (7, 7), 2.0)
                        mask = torch.from_numpy(mask_np).to(device)
                        
                        if pass_idx == num_passes - 1 and frame_idx % 10 == 0:
                            utils.save_image(
                                mask_np,
                                os.path.join(debug_dir, f"frame_{frame_idx:04d}_mask_pass{pass_idx+1}.png")
                            )
                        
                        # Warp previous stylized frame
                        prev_stylized = torch.clamp(prev_stylized, 0.0, 1.0)
                        warped_prev = video_utils.warp_image(
                            prev_stylized,
                            flow_forward
                        ).to(device)
                        warped_prev = torch.clamp(warped_prev, 0.0, 1.0)
                        
                        if pass_idx == num_passes - 1 and frame_idx % 10 == 0:
                            utils.save_image(
                                warped_prev,
                                os.path.join(debug_dir, f"frame_{frame_idx:04d}_warped_pass{pass_idx+1}.png")
                            )
                        
                        # Blend with adaptive alpha based on pass number
                        alpha = max(0.7, 0.95 - 0.05 * pass_idx)  # Less aggressive blending
                        mask = mask.unsqueeze(0).unsqueeze(0)
                        current_stylized = (
                            alpha * current_stylized + 
                            (1 - alpha) * warped_prev * mask + 
                            current_stylized * (1 - mask)
                        )
                        
                        current_stylized = torch.clamp(current_stylized, 0.0, 1.0)
                    
                    # Save debug outputs for last pass
                    if pass_idx == num_passes - 1 and frame_idx % 10 == 0:
                        utils.save_image(
                            current_stylized,
                            os.path.join(debug_dir, f"frame_{frame_idx:04d}_final_pass{pass_idx+1}.png")
                        )

                    # Update previous frame references
                    prev_frame = current_frame
                    prev_stylized = current_stylized
                    
                    # Convert and store frame
                    frame_to_append = (current_stylized.squeeze(0).cpu().detach() * 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    current_pass_frames.append(frame_to_append)
        
        # Update stylized frames after each pass
        stylized_frames = current_pass_frames
    
    # Save final video with unique name
    output_filename = f"stylized_multipass_{unique_prefix}.mp4"
    output_path = os.path.join(video_config['output_path'], output_filename)
    utils.save_video(stylized_frames, output_path, fps=30)
    
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
    multi_pass_stylize(
        video_config,
        "test-video-1.mp4",
        num_passes=3,
        debug_dir="debug_output"
    )