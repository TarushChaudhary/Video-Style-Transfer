import os
import sys
from config import get_video_config
from inference_image import stylize_single_image
from main import extract_and_process_frames
from single_pass_tc import single_pass_stylize
from multipass_tc import multi_pass_stylize
import torch
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('style_transfer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Video and model paths
videos = ["data/content-videos/test-video-1.mp4", 
          "data/content-videos/test-video-2.mp4", 
          "data/content-videos/test-video-3.mp4"]
model_path = "models/binaries/"
models = ["mosaic.pth", "starry.pth", "colorful.pth", "fuzzy.pth"]

def validate_paths():
    """Validate existence of all required files and directories"""
    try:
        # Check videos
        for video in videos:
            if not os.path.exists(video):
                raise FileNotFoundError(f"Video file not found: {video}")
        
        # Check models
        for model in models:
            if not os.path.exists(model_path + model):
                raise FileNotFoundError(f"Model file not found: {model}")
        
        # Create necessary directories
        os.makedirs("output-stylized-videos", exist_ok=True)
        os.makedirs("debug_output", exist_ok=True)
        
        return True
    except Exception as e:
        logging.error(f"Path validation failed: {str(e)}")
        return False

def run_simple_video(video_path, model):
    """Run simple video stylization"""
    try:
        logging.info(f"Processing simple video: {video_path} with model: {model}")
        extract_and_process_frames(
            video_path, 
            "output-stylized-videos", 
            target_resolution=(500, 500), 
            model=model
        )
        logging.info(f"Successfully processed {video_path} with {model}")
    except Exception as e:
        logging.error(f"Error in simple video processing - {video_path} with {model}: {str(e)}")
        raise

def run_single_pass_video(video_path, model, file_name):
    """Run single pass video stylization"""
    try:
        logging.info(f"Processing single pass video: {video_path} with model: {model}")
        video_config = get_video_config()
        os.makedirs(video_config['output_path'], exist_ok=True)
        video_config['input_path'] = video_path
        video_config['model_name'] = model
        single_pass_stylize(video_config, file_name, debug_dir="debug_output_single_pass")
        logging.info(f"Successfully processed single pass for {video_path} with {model}")
    except Exception as e:
        logging.error(f"Error in single pass processing - {video_path} with {model}: {str(e)}")
        raise

def run_multi_pass_video(video_path, model, file_name):
    """Run multi pass video stylization"""
    try:
        logging.info(f"Processing multi pass video: {video_path} with model: {model}")
        video_config = get_video_config()
        os.makedirs(video_config['output_path'], exist_ok=True)
        video_config['input_path'] = video_path
        video_config['model_name'] = model
        multi_pass_stylize(video_config, file_name, num_passes=3, debug_dir="debug_output_multi_pass")
        logging.info(f"Successfully processed multi pass for {video_path} with {model}")
    except Exception as e:
        logging.error(f"Error in multi pass processing - {video_path} with {model}: {str(e)}")
        raise

def main():
    try:
        # Check CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Validate paths before processing
        if not validate_paths():
            logging.error("Path validation failed. Exiting.")
            return

        # Process simple videos
        for video in videos:
            for model in models:
                try:
                    torch.cuda.empty_cache()
                    run_simple_video(video, model)
                except Exception as e:
                    logging.error(f"Failed simple video processing for {video} with {model}")
                    continue

        # Process single pass videos
        for video in videos:
            for model in models:
                try:
                    torch.cuda.empty_cache()
                    run_single_pass_video(video, model, video.split("/")[-1])
                except Exception as e:
                    logging.error(f"Failed single pass processing for {video} with {model}")
                    continue

        # Process multi pass videos
        for video in videos:
            for model in models:
                try:
                    torch.cuda.empty_cache()
                    run_multi_pass_video(video, model, video.split("/")[-1])
                except Exception as e:
                    logging.error(f"Failed multi pass processing for {video} with {model}")
                    continue

    except Exception as e:
        logging.error(f"Main process failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Process failed with error: {str(e)}")
        sys.exit(1)

