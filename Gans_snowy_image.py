import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
#from gan_networks import define_G
from lib.gan_networks import define_G



# Function to map resize method
def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS}
    return mapper[method]


# Function to scale image width while maintaining aspect ratio
def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


# Function to get image transformation
def get_transform(load_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    transform_list = [transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


# Function to convert tensor to image
def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: transpose and scale
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


# Define the model and transformations
def create_model_and_transform(pretrained="C:/Users/musta/.cache/torch/hub/checkpoints/clear2snowy.pth"):
    # Model parameters
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'resnet_9blocks'
    norm = 'instance'
    no_dropout = True
    init_type = 'normal'
    init_gain = 0.02
    gpu_ids = []

    # Define the generator model
    netG_A = define_G(input_nc, output_nc, ngf, netG, norm, not no_dropout, init_type, init_gain, gpu_ids)

    # Load the pre-trained model weights
    if pretrained:
        chkpntA = torch.load(pretrained)
        netG_A.load_state_dict(chkpntA)
    netG_A.eval()

    # Creating transformation pipeline
    load_size = 1280
    crop_size = 224
    image_transforms = get_transform(load_size=load_size, crop_size=crop_size)
    return netG_A, image_transforms


# Run inference on the image
def run_inference(img_path, model, transform):
    image = Image.open(img_path)
    inputs = transform(image).unsqueeze(0)  # Convert to tensor and add batch dimension

    with torch.no_grad():
        out = model(inputs)
    out = tensor2im(out)
    return Image.fromarray(out)


# Main function to process images
def process_images(input_folder, output_folder, model, transform):
    image_files = sorted(os.listdir(input_folder))  # List image files in the input folder
    for img in tqdm(image_files):
        img_path = os.path.join(input_folder, img)
        output_path = os.path.join(output_folder, img.split('.')[0] + "-gan.jpg")
        if not os.path.exists(output_path):  # Skip if output file already exists
            out = run_inference(img_path=img_path, model=model, transform=transform)
            out.save(output_path)


if __name__ == "__main__":
    # Define paths
    input_folder = "images"  # Folder where clear images are stored
    output_folder = "output"  # Folder where generated images will be saved

    # Create model and transformation pipeline
    gan, image_transforms = create_model_and_transform(pretrained="C:/Users/musta/.cache/torch/hub/checkpoints/clear2snowy.pth")

    # Process the images and generate snowy images
    process_images(input_folder, output_folder, gan, image_transforms)
