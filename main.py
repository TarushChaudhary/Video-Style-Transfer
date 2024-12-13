import os
import cv2
import numpy as np
from tqdm import tqdm
from inference_image import stylize_static_image
from video_stylize import process_video
from config import get_inference_config, get_video_config
import trainingdata.utils as utils

def extract_and_process_frames(video_path, output_dir, max_seconds=8, target_fps=30, target_resolution=(500, 500), model="starry.pth"):
    """
    Extract frames from video, save them, and process them with both methods
    """
    # Extract video name and model name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = os.path.splitext(os.path.basename(model))[0]
    
    # Create unique directory names incorporating both video and model names
    frames_dir = os.path.join('data/content-images', f'{video_name}_{model_name}')
    stylized_frames_dir = os.path.join('data/output-stylized-images', f'{video_name}_{model_name}')
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(stylized_frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Read video
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frames to extract
    frames_to_extract = min(int(max_seconds * target_fps), frame_count)
    frame_interval = original_fps / target_fps

    # Get inference configuration
    inference_config = get_inference_config()
    inference_config['content_images_path'] = frames_dir
    inference_config['output_images_path'] = stylized_frames_dir
    inference_config['should_not_display'] = True
    inference_config['model_name'] = model

    # Extract and save frames
    print("Extracting frames...")
    frame_paths = []
    current_frame = 0
    frame_counter = 0

    while frame_counter < frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame))
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to target resolution
        frame = cv2.resize(frame, target_resolution)

        # Save frame
        frame_path = os.path.join(frames_dir, f'frame_{frame_counter:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        
        frame_counter += 1
        current_frame += frame_interval

    cap.release()

    # Process frames using inference_image.py (frame by frame)
    print("\nProcessing frames individually...")
    for frame_path in tqdm(frame_paths):
        inference_config['content_input'] = os.path.basename(frame_path)
        stylize_static_image(inference_config)

    # Create video from individually stylized frames with unique name
    output_video_name = f'stylized_{video_name}_{model_name}.mp4'
    print(f"\nCreating video: {output_video_name}")
    frame_to_video(
        stylized_frames_dir,
        os.path.join(output_dir, output_video_name),
        target_fps
    )


def frame_to_video(frames, output_path, fps):
    """
    Convert frames to a video. Accepts either:
    - A directory path containing frame images
    - A list of torch tensors containing frame data
    """
    if isinstance(frames, (str, bytes, os.PathLike)):
        # Handle directory of image files
        frame_files = sorted([f for f in os.listdir(frames) if f.endswith('.jpg') or f.endswith('.png')])
        if not frame_files:
            print("No frames found to create video")
            return

        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(frames, frame_files[0]))
        height, width = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Add frames to video
        for frame_name in frame_files:
            frame_path = os.path.join(frames, frame_name)
            frame = cv2.imread(frame_path)
            out.write(frame)

    else:
        # Handle list of tensor frames
        if not frames:
            print("No frames provided to create video")
            return

        # Get dimensions from first frame
        height, width = frames[0].shape[1:3]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Add frames to video
        for frame in frames:
            # Convert tensor to numpy array and proper format for cv2
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

    out.release()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/content-videos', exist_ok=True)
    os.makedirs('data/content-images', exist_ok=True)
    os.makedirs('data/output-stylized-images', exist_ok=True)
    os.makedirs('data/output-stylized-videos', exist_ok=True)

    # Process video
    video_path = os.path.join('data/content-videos', 'test-video-2.mp4')
    output_dir = 'data/output-stylized-videos'
    
    if not os.path.exists(video_path):
        print(f"Please place your input video as 'input.mp4' in {os.path.dirname(video_path)}")
    else:
        extract_and_process_frames(video_path, output_dir, target_resolution=(500, 500))
