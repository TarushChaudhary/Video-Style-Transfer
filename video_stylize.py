import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from TransformerModel.model import TransformerNet
import utils
import video_utils
from config import get_video_config

def process_video(video_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = TransformerNet().to(device)
    state_dict = torch.load(os.path.join(
        video_config['model_binaries_path'], 
        video_config['model_name']
    ))['state_dict']
    model.load_state_dict(state_dict)
    
    # Process video
    video_path = os.path.join(video_config['input_path'], 'input.mp4')
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new dimensions maintaining aspect ratio
    new_width = video_config['img_width']
    new_height = int(height * (new_width / width))
    
    # Setup video writer
    output_path = os.path.join(video_config['output_path'], 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Process frames
    prev_stylized = None
    frames = []
    
    # Read all frames
    print("Reading frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (new_width, new_height))
        frames.append(frame)
    
    # Process frames with temporal consistency
    print("Processing frames...")
    for i in tqdm(range(len(frames))):
        current_frame = frames[i]
        
        # Prepare current frame
        current_tensor = utils.prepare_img(
            current_frame, 
            target_shape=(new_height, new_width),
            device=device,
            should_normalize=True
        )
        
        if prev_stylized is not None:
            # Estimate optical flow
            flow_forward = video_utils.estimate_optical_flow(
                frames[i-1], current_frame
            ).to(device)
            flow_backward = video_utils.estimate_optical_flow(
                current_frame, frames[i-1]
            ).to(device)
            
            # Compute consistency mask
            mask = video_utils.compute_consistency_mask(
                flow_forward, flow_backward
            ).to(device)
            
            # Warp previous stylized frame
            warped_prev = video_utils.warp_image(
                prev_stylized, 
                flow_forward
            )
            
            # Initialize current frame with warped previous frame
            current_stylized = model(current_tensor)
            
            # Apply temporal consistency
            temporal_loss = video_utils.temporal_consistency_loss(
                current_stylized, 
                warped_prev, 
                mask
            )
            
            # Blend frames based on temporal consistency
            alpha = 0.8
            current_stylized = (alpha * current_stylized + 
                              (1 - alpha) * warped_prev)
        else:
            current_stylized = model(current_tensor)
        
        # Save for next iteration
        prev_stylized = current_stylized.detach()
        
        # Convert to image and save
        stylized_frame = utils.post_process_image(
            current_stylized[0].cpu().numpy()
        )
        out.write(cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR))
    
    # Cleanup
    cap.release()
    out.release()




def process_frame(frame, prev_stylized, model, temporal_weight):
    """Process a single frame with style transfer and temporal consistency"""
    device = frame.device
    
    # Initialize with previous frame if available
    if prev_stylized is not None:
        flow = video_utils.estimate_optical_flow(
            frame.cpu().numpy(),
            prev_stylized.cpu().numpy()
        ).to(device)
        
        # Compute consistency mask
        backward_flow = video_utils.estimate_optical_flow(
            prev_stylized.cpu().numpy(),
            frame.cpu().numpy()
        ).to(device)
        mask = video_utils.compute_consistency_mask(flow, backward_flow).to(device)
        
        # Warp previous stylized frame
        warped_prev = video_utils.warp_image(prev_stylized.unsqueeze(0), flow).squeeze(0)
        
        # Apply style transfer with temporal consistency
        current_stylized = model(frame.unsqueeze(0))
        temporal_loss = video_utils.temporal_consistency_loss(
            current_stylized,
            warped_prev,
            mask
        ) * temporal_weight
        
        # Blend current and warped previous frame
        alpha = 0.8
        stylized = alpha * current_stylized + (1 - alpha) * warped_prev
    else:
        # First frame: just apply style transfer
        stylized = model(frame.unsqueeze(0))
    
    return stylized.squeeze(0)

def process_video_multipass(video_config, num_passes=15):
    """Multi-pass video processing """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = TransformerNet().to(device)
    state_dict = torch.load(os.path.join(
        video_config['model_binaries_path'], 
        video_config['model_name']
    ))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    # Read video frames
    video_path = os.path.join(video_config['input_path'], 'input.mp4')
    frames = video_utils.read_video_frames(video_path)
    frames = [torch.from_numpy(frame).float().to(device) for frame in frames]
    
    # Initialize storage for stylized frames
    stylized_frames = [None] * len(frames)
    
    for pass_idx in range(num_passes):
        print(f"Processing pass {pass_idx + 1}/{num_passes}")
        
        # Alternate between forward and backward passes
        forward_pass = (pass_idx % 2 == 0)
        
        # Process frames in alternating order
        frame_indices = range(len(frames))
        if not forward_pass:
            frame_indices = reversed(frame_indices)
        
        prev_stylized = None
        next_stylized = None
        
        for i in frame_indices:
            # Get neighboring stylized frames from previous pass
            if pass_idx > 0:
                if i > 0:
                    prev_stylized = stylized_frames[i-1]
                if i < len(frames) - 1:
                    next_stylized = stylized_frames[i+1]
            
            # Blend neighboring frames
            blend_weight = video_utils.compute_blend_weight(pass_idx, num_passes)
            current_frame = video_utils.blend_neighboring_frames(
                frames[i],
                prev_stylized,
                next_stylized,
                blend_weight
            )
            
            # Apply style transfer with temporal consistency
            temporal_weight = video_utils.get_temporal_weight(pass_idx)
            stylized = process_frame(
                current_frame,
                prev_stylized,
                model,
                temporal_weight
            )
            
            # Save stylized frame
            stylized_frames[i] = stylized
            
            # Save intermediate result
            if video_config.get('save_intermediate', False):
                output_path = os.path.join(
                    video_config['output_path'],
                    f'pass_{pass_idx:02d}_frame_{i:04d}.png'
                )
                utils.save_image(stylized, output_path)
    
    # Create final video
    output_path = os.path.join(video_config['output_path'], 'output.mp4')
    video_utils.create_video(stylized_frames, output_path, fps=30)
    
    return stylized_frames

if __name__ == "__main__":
    video_config = get_video_config()
    os.makedirs(video_config['output_path'], exist_ok=True)
    process_video(video_config) 