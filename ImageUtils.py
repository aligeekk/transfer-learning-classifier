from torchvision import datasets, transforms
from torch import utils
from argparse import ArgumentTypeError
import numpy as np

data_transforms = {
    'training': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize(255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'default': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def generateDataLoaderDictionary(data_dir):
    """data_dir is assumed to contain three sub-directories '/train', '/valid', '/test'"""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, data_transforms['default']),
        'testing': datasets.ImageFolder(test_dir, data_transforms['default']),
    }
    dataloaders = {
        'training': utils.data.DataLoader(image_datasets['training'], batch_size = 64, shuffle = True),
        'validation': utils.data.DataLoader(image_datasets['validation'], batch_size = 64),
        'testing': utils.data.DataLoader(image_datasets['testing'], batch_size = 64),
    }
    return image_datasets, dataloaders

def check_positive_integral_value(input_value):
    try:
        int_input_value = int(input_value)
        if int_input_value <= 0:
            raise ArgumentTypeError("{} is not a positive integer".format(int_input_value))
        return int_input_value
    except:
        raise ArgumentTypeError("{} is not a positive integer".format(input_value))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    old_size = image.size
    scale_ratio = float(256)/min(old_size)
    new_size = tuple([int(x*scale_ratio) for x in old_size])
    new_width, new_height = new_size
    resized_im = image.resize(new_size)
    cropped_im = resized_im.crop((float(new_width - 224) / 2, float(new_height - 224) / 2,
                                  new_width - float(new_width - 224) / 2, new_height - float(new_height - 224) / 2))
    np_image = np.array(cropped_im)
    np_image_scaled = np_image / 255
    color_mean = np.array([0.485, 0.456, 0.406])
    color_std = np.array([0.229, 0.224, 0.225])
    np_image_standardized = (np_image_scaled - color_mean) / color_std
    np_image_standardized.transpose((2,0,1))
    return np.array(np_image_standardized)
