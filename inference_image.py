import os
import torch
from torch.utils.data import DataLoader
import trainingdata.utils as utils
from TransformerModel.model import TransformerNet
from config import get_inference_config


def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]))
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    if inference_config['verbose']:
        utils.print_model_metadata(training_state)

    with torch.no_grad():
        if os.path.isdir(inference_config['content_input']):  # do a batch stylization
            img_dataset = utils.SimpleDataset(inference_config['content_input'], inference_config['img_width'])
            img_loader = DataLoader(img_dataset, batch_size=inference_config['batch_size'])

            try:
                processed_imgs_cnt = 0
                for batch_id, img_batch in enumerate(img_loader):
                    processed_imgs_cnt += len(img_batch)
                    if inference_config['verbose']:
                        print(f'Processing batch {batch_id + 1} ({processed_imgs_cnt}/{len(img_dataset)} processed images).')

                    img_batch = img_batch.to(device)
                    stylized_imgs = stylization_model(img_batch).to('cpu').numpy()
                    for stylized_img in stylized_imgs:
                        utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=False)
            except Exception as e:
                print(e)
                print(f'Consider making the batch_size (current = {inference_config["batch_size"]} images) or img_width (current = {inference_config["img_width"]} px) smaller')
                exit(1)

        else:  # do stylization for a single image
            content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content_input'])
            content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
            stylized_img = stylization_model(content_image).to('cpu').numpy()[0]
            utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=inference_config['should_not_display'])


def stylize_single_image(input_file, model_name):    # Get configuration from config.py
    inference_config = get_inference_config()
    
    assert utils.dir_contains_only_models(inference_config['model_binaries_path']), f'Model directory should contain only model binaries.'
    os.makedirs(inference_config['output_images_path'], exist_ok=True)
    inference_config['model_name'] = model_name
    inference_config['content_input'] = input_file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]))
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    if inference_config['verbose']:
        utils.print_model_metadata(training_state)

    with torch.no_grad():
        content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content_input'])
        content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
        stylized_img = stylization_model(content_image).to('cpu').numpy()[0]
        utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=inference_config['should_not_display'])