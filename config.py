import os

# Fixed configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
STYLE_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'style-images')
MODEL_BINARIES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
CHECKPOINTS_ROOT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
IMAGE_SIZE = 256  # training images from MS COCO are resized to image_size x image_size
BATCH_SIZE = 4

# Training configuration
STYLE_IMG_NAME = 'edtaonisl.jpg'
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
