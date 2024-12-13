import os
import cv2
import torch
import numpy as np
import video_utils
import trainingdata.utils as utils

def test_optical_flow(video_path, output_dir="flow_test_output"):
    """
    Test optical flow computation between first two frames of a video.
    Saves visualizations of flow in x and y directions, and flow magnitude.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read first two frames from video
    cap = cv2.VideoCapture(video_path)
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()
    
    if not (ret1 and ret2):
        print("Failed to read frames from video!")
        return
    
    # Convert to RGB
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # Resize frames if needed (optional, adjust size as needed)
    target_size = 512
    frame1 = cv2.resize(frame1, (target_size, target_size))
    frame2 = cv2.resize(frame2, (target_size, target_size))
    
    # Compute flow
    print("Computing forward flow...")
    flow_forward = video_utils.estimate_optical_flow(frame1, frame2)
    print("Computing backward flow...")
    flow_backward = video_utils.estimate_optical_flow(frame2, frame1)
    
    # Convert flow to numpy for visualization
    flow_forward_np = flow_forward.numpy()
    
    # Save flow visualizations
    # Flow in x direction
    utils.save_image(
        normalize_flow(flow_forward_np[0]),
        os.path.join(output_dir, "flow_x.png")
    )
    
    # Flow in y direction
    utils.save_image(
        normalize_flow(flow_forward_np[1]),
        os.path.join(output_dir, "flow_y.png")
    )
    
    # Flow magnitude
    flow_magnitude = np.sqrt(np.sum(flow_forward_np**2, axis=0))
    utils.save_image(
        normalize_flow(flow_magnitude),
        os.path.join(output_dir, "flow_magnitude.png")
    )
    
    # Compute and save consistency mask
    print("Computing consistency mask...")
    mask = video_utils.compute_consistency_mask(flow_forward, flow_backward)
    utils.save_image(
        mask,
        os.path.join(output_dir, "consistency_mask.png")
    )
    
    # Test warping
    print("Testing image warping...")
    warped_frame2 = video_utils.warp_image(
        torch.from_numpy(frame2).float().permute(2, 0, 1) / 255.0,
        flow_forward
    )
    utils.save_image(
        warped_frame2,
        os.path.join(output_dir, "warped_image.png")
    )
    
    # Save input frames for reference
    utils.save_image(frame1, os.path.join(output_dir, "frame1.png"))
    utils.save_image(frame2, os.path.join(output_dir, "frame2.png"))
    
    print(f"Results saved to {output_dir}/")
    return flow_forward, flow_backward, mask

def normalize_flow(flow):
    """Normalize flow values to [0, 1] range"""
    flow_min = flow.min()
    flow_max = flow.max()
    if flow_max - flow_min > 0:
        return (flow - flow_min) / (flow_max - flow_min)
    return flow

if __name__ == "__main__":
    video_path = "data/content-videos/test-video-2.mp4" 
    output_dir = "flow_test_output-2"
    
    test_optical_flow(video_path, output_dir) 