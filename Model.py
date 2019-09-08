from ImageUtils.py import generateDataLoaderDictionary
from torchvision import models
from torch import nn, device, cuda, optim

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

def update_from_training_data(model, dataloaders, device, criterion, optimizer):
    running_loss = 0
    for images, labels in dataloaders['training']: # should be batch of 32 images with labels in each iteration
        images_cuda, labels_cuda = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images_cuda), labels_cuda)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss

def evaluate_model_on_validation():
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
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    train_losses.append(running_loss / len(dataloaders['training']))
    validation_losses.append(validation_loss / len(dataloaders['validation']))
    model.train()


def run_feed_forward_back_propagation(model, epochs, learning_rate, dataloaders, criterion):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    train_losses = []
    validation_losses = []

    for e in range(epochs):
        running_loss = update_from_training_data(model, dataloaders, device, criterion, optimizer)
        validation_loss, validation_accuracy = evaluate_model_on_validation()





def train_and_save_model(data_directory, save_directory, architecture,
                         learning_rate, hidden_units, epochs, is_gpu_enabled):
    dataloaders = generateDataLoaderDictionary(data_directory)
    model = models[architecture](pretrained=True)

    freeze_model_parameters(model)
    model_classifier = generate_classifier(model.classifier[0].in_features, hidden_units)
    model.classifier = model_classifier
    criterion = nn.NLLLoss()
    device = torch.device("cuda:0" if (is_gpu_enabled and torch.cuda.is_available()) else "cpu")

    run_feed_forward_back_propagation(model, epochs, learning_rate, dataloaders, criterion)
