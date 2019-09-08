from ImageUtils import generateDataLoaderDictionary
from torchvision import models
import torch
from torch import nn, device, cuda, optim, no_grad
from workspace-utils import keep_awake

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def generate_classifier(classifier_input_size, hidden_units):
    flower_classifier = nn.Sequential(OrderedDict([
        ('hiddenA', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('hiddenB', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
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
    with no_grad():
        for images, labels in dataloaders['validation']:
            images_test_cuda, labels_test_cuda = images.to(device), labels.to(device)

            log_forward_pass = model(images_test_cuda)
            validation_loss += criterion(log_forward_pass, labels_test_cuda).item()
            forward_pass = exp(log_forward_pass)
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
    with no_grad():
        for images, labels in dataloaders['testing']:
            images_test_cuda, labels_test_cuda = images.to(device), labels.to(device)

            test_log_forward_pass = model(images_test_cuda)
            test_loss += criterion(test_log_forward_pass, labels_test_cuda).item()
            test_forward_pass = exp(test_log_forward_pass)
            top_p, top_class = test_forward_pass.topk(1, dim=1)

            test_equals = top_class == labels_test_cuda.view(*top_class.shape)
            test_accuracy += torch.mean(test_equals.type(torch.FloatTensor)).item()
    model.train()
    return test_loss, test_accuracy

def run_feed_forward_back_propagation(model, epochs, learning_rate, dataloaders, criterion):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
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

def train_and_save_model(data_directory, save_directory, architecture,
                         learning_rate, hidden_units, epochs, is_gpu_enabled):
    image_datasets, dataloaders = generateDataLoaderDictionary(data_directory)
    model = models[architecture](pretrained=True)

    freeze_model_parameters(model)
    model_classifier = generate_classifier(model.classifier[0].in_features, hidden_units)
    model.classifier = model_classifier
    criterion = nn.NLLLoss()
    device = torch.device("cuda:0" if (is_gpu_enabled and torch.cuda.is_available()) else "cpu")

    run_feed_forward_back_propagation(model, epochs, learning_rate, dataloaders, criterion)
    save_to_checkpoint(model, save_directory, architecture, epochs, image_datasets)
