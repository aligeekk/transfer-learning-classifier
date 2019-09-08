from torchvision import datasets, transforms
from torch import utils
from argparse import ArgumentTypeError

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
