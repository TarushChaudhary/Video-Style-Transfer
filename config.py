import os
# TRAINING PARAMETERS
# Fixed configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
STYLE_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'style')
MODEL_BINARIES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
CHECKPOINTS_ROOT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
IMAGE_SIZE = 256  # training images from MS COCO are resized to image_size x image_size
BATCH_SIZE = 4

# Training configuration
STYLE_IMG_NAME = 'train_3.jpeg'
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 4e5
TV_WEIGHT = 0
NUM_OF_EPOCHS = 2
SUBSET_SIZE = None  # number of MS COCO images to use, None means all (~83k)

# Logging configuration
ENABLE_TENSORBOARD = True
IMAGE_LOG_FREQ = 100
CONSOLE_LOG_FREQ = 500
CHECKPOINT_FREQ = 2000


# INFERENCE PARAMETERS
CONTENT_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'content-images')
OUTPUT_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'output-images')
CONTENT_INPUT = 'taj_mahal.jpg'  # Content image(s) to stylize
INFERENCE_BATCH_SIZE = 5  # Batch size for directory processing
INFERENCE_IMG_WIDTH = 500  # Resize content image to this width
INFERENCE_MODEL_NAME = 'mosaic_4e5_e2.pth'  # Model binary to use for stylization
SHOULD_DISPLAY = True  # Should display the stylized result
VERBOSE = False  # Print model metadata
REDIRECTED_OUTPUT = None  # Overwrite default output dir

# VIDEO PARAMETERS
VIDEO_INPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'content-videos')
VIDEO_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'output-videos')
TEMPORAL_WEIGHT = 1e4  # Weight for temporal consistency loss
FLOW_WEIGHT = 1.0  # Weight for optical flow consistency
MULTI_PASS = True  # Whether to use multi-pass processing
NUM_PASSES = 2  # Number of forward/backward passes

def get_training_config():
    """Returns a dictionary containing all configuration parameters"""
    checkpoints_path = os.path.join(CHECKPOINTS_ROOT_PATH, STYLE_IMG_NAME.split('.')[0])
    if CHECKPOINT_FREQ is not None:
        os.makedirs(checkpoints_path, exist_ok=True)
    
    return {
        'dataset_path': DATASET_PATH,
        'style_images_path': STYLE_IMAGES_PATH,
        'model_binaries_path': MODEL_BINARIES_PATH,
        'checkpoints_path': checkpoints_path,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'style_img_name': STYLE_IMG_NAME,
        'content_weight': CONTENT_WEIGHT,
        'style_weight': STYLE_WEIGHT,
        'tv_weight': TV_WEIGHT,
        'num_of_epochs': NUM_OF_EPOCHS,
        'subset_size': SUBSET_SIZE,
        'enable_tensorboard': ENABLE_TENSORBOARD,
        'image_log_freq': IMAGE_LOG_FREQ,
        'console_log_freq': CONSOLE_LOG_FREQ,
        'checkpoint_freq': CHECKPOINT_FREQ,
    }

def get_inference_config():
    """Returns a dictionary containing all inference configuration parameters"""


    return {
        'content_images_path': CONTENT_IMAGES_PATH,
        'output_images_path': OUTPUT_IMAGES_PATH,
        'content_input': CONTENT_INPUT,
        'batch_size': INFERENCE_BATCH_SIZE,
        'img_width': INFERENCE_IMG_WIDTH,
        'model_name': INFERENCE_MODEL_NAME,
        'should_not_display': not SHOULD_DISPLAY,
        'verbose': VERBOSE,
        'redirected_output': REDIRECTED_OUTPUT,
        'model_binaries_path': MODEL_BINARIES_PATH
    }

def get_video_config():
    """Returns a dictionary containing all video processing configuration parameters"""
    return {
        'input_path': VIDEO_INPUT_PATH,
        'output_path': VIDEO_OUTPUT_PATH,
        'temporal_weight': TEMPORAL_WEIGHT,
        'flow_weight': FLOW_WEIGHT,
        'multi_pass': MULTI_PASS,
        'num_passes': NUM_PASSES,
        'model_binaries_path': MODEL_BINARIES_PATH,
        'model_name': INFERENCE_MODEL_NAME,
        'img_width': INFERENCE_IMG_WIDTH,
    }