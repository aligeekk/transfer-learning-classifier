from ImageUtils import generateDataLoaderDictionary
from torchvision import models
from workspace_utils import keep_awake
from ImageUtils import process_image
from collections import OrderedDict
import torch
import numpy as np

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def generate_classifier(classifier_input_size, hidden_units):
    flower_classifier = torch.nn.Sequential(OrderedDict([
        ('hiddenA', torch.nn.Linear(classifier_input_size, hidden_units)),
        ('relu', torch.nn.ReLU()),
        ('dropout', torch.nn.Dropout(p=0.2)),
        ('hiddenB', torch.nn.Linear(hidden_units, 102)),
        ('output', torch.nn.LogSoftmax(dim=1))
    ]))
    return flower_classifier

def update_from_training_data(model, dataloaders, device, criterion, optimizer, train_losses):
    running_loss = 0
    for images, labels in dataloaders['training']: # should be batch of 32 images with labels in each iteration
        images_cuda, labels_cuda = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images_cuda), labels_cuda)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_losses.append(running_loss / len(dataloaders['training']))
    return running_loss

def evaluate_model_on_validation(model, dataloaders, device, criterion, validation_losses):
    validation_loss = 0
    validation_accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders['validation']:
            images_test_cuda, labels_test_cuda = images.to(device), labels.to(device)

            log_forward_pass = model(images_test_cuda)
            validation_loss += criterion(log_forward_pass, labels_test_cuda).item()
            forward_pass = torch.exp(log_forward_pass)
            top_p, top_class = forward_pass.topk(1, dim=1)

            equals = top_class == labels_test_cuda.view(*top_class.shape)
            validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    validation_losses.append(validation_loss / len(dataloaders['validation']))
    model.train()
    return validation_loss, validation_accuracy

def evaluate_model_on_testing(model, dataloaders, device, criterion):
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders['testing']:
            images_test_cuda, labels_test_cuda = images.to(device), labels.to(device)

            test_log_forward_pass = model(images_test_cuda)
            test_loss += criterion(test_log_forward_pass, labels_test_cuda).item()
            test_forward_pass = torch.exp(test_log_forward_pass)
            top_p, top_class = test_forward_pass.topk(1, dim=1)

            test_equals = top_class == labels_test_cuda.view(*top_class.shape)
            test_accuracy += torch.mean(test_equals.type(torch.FloatTensor)).item()
    model.train()
    return test_loss, test_accuracy

def run_feed_forward_back_propagation(model, epochs, learning_rate, dataloaders, criterion, device):
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    train_losses = []
    validation_losses = []

    for e in keep_awake(range(epochs)):
        running_loss = update_from_training_data(model, dataloaders, device, criterion, optimizer, train_losses)
        validation_loss, validation_accuracy = evaluate_model_on_validation(model, dataloaders, device, criterion, validation_losses)
        test_loss, test_accuracy = evaluate_model_on_testing(model, dataloaders, device, criterion)
        print("Epoch: {}/{}".format(e+1, epochs))
        print("Training Loss: {:.3f}..".format(running_loss/len(dataloaders['training'])))
        print("Validation Loss: {:.3f}..".format(validation_loss/len(dataloaders['validation'])))
        print("Validation Accuracy: {:.3f}..".format(validation_accuracy/len(dataloaders['validation'])))
        print("Testing Loss: {:.3f}..".format(test_loss/len(dataloaders['testing'])))
        print("Test Accuracy: {:.3f}..\n".format(test_accuracy/len(dataloaders['testing'])))

def save_to_checkpoint(model, save_directory, architecture, optimizer, epochs, image_datasets):
    checkpoint = {
        'input_size': 50176,
        'output_size': 102,
        'classifier': model.classifier,
        'state_dict': model.classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'model_architecture': architecture,
        'class_to_idx': image_datasets['training'].class_to_idx,
    }
    torch.save(checkpoint, save_directory)

def get_model(architecture):
    """The default model is vgg11."""
    if architecture == 'resnet18':
        return models.resnet18(pretrained=True)
    elif architecture == 'squeezenet1_0':
        return models.squeezenet1_0(pretrained=True)
    elif architecture == 'densenet121':
        return models.densenet121(pretrained=True)
    elif architecture == 'vgg13':
        return models.vgg13(pretrained=True)
    elif architecture == 'vgg16':
        return models.vgg16(pretrained=True)
    elif architecture == 'vgg19':
        return models.vgg19(pretrained=True)
    else:
        return models.vgg11(pretrained=True)

def train_and_save_model(data_directory, save_directory, architecture,
                         learning_rate, hidden_units, epochs, is_gpu_enabled):
    image_datasets, dataloaders = generateDataLoaderDictionary(data_directory)
    model = get_model(architecture)
    freeze_model_parameters(model)
    model_classifier = generate_classifier(model.classifier[0].in_features, hidden_units)
    model.classifier = model_classifier
    criterion = torch.nn.NLLLoss()
    device = torch.device("cuda:0" if (is_gpu_enabled and torch.cuda.is_available()) else "cpu")

    run_feed_forward_back_propagation(model, epochs, learning_rate, dataloaders, criterion, device)
    save_to_checkpoint(model, save_directory, architecture, epochs, image_datasets)

def load_checkpoint(pathname, device):
    try:
        checkpoint = torch.load(pathname)
    except err:
        print(err)
        print("{} is not a valid path".format(pathname))
    else:
        try:
            model = models[checkpoint['model_architecture']](pretrained=True)
        except err:
            print(err)
            print("{} is not a valid model architecture".format(checkpoint['model_architecture']))
        else:
            model.to(device)
            # Freeze VGG network pre-trained parameters
            for param in model.parameters():
                param.requires_grad = False
            model.class_to_idx = checkpoint['class_to_idx']
            model.classifier = checkpoint['classifier']
            return model

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    im = Image.open(image_path)
    formatted_image = process_image(im)
    fi_to_torch = torch.tensor(np.array([formatted_image.transpose((2,0,1))])).type(torch.FloatTensor)
    model.eval()
    with torch.no_grad():
        probability_results = model(fi_to_torch.to(device))
    model.train()
    probs, classes = probability_results.topk(topk)
    return torch.exp(probs), classes

def make_prediction(path_to_image, checkpoint, top_k, category_names, is_gpu_enabled):
    device = torch.device("cuda:0" if (is_gpu_enabled and torch.cuda.is_available()) else "cpu")
    model = load_checkpoint(checkpoint, device)
    probs, classes = predict(path_to_image, model, top_k, device)
    return probs, classes
