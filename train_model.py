import os
import config
from train import train
from config import get_training_config
from trainingdata.get_data import check_for_msco_data


if __name__ == "__main__":
    check_for_msco_data()
    style_images = os.listdir(config.STYLE_IMAGES_PATH)
    for style_image in style_images:
        training_config = get_training_config()
        training_config['style_img_name'] = style_image
        train(training_config)

