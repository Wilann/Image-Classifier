# Train a new network  on a dataset and save the model as a checkpoint
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import json
from time import time

# Initialize argparse object
parser = argparse.ArgumentParser(description="Parser of training script")

# Define mandatory argumentd
parser.add_argument('data_dir', help='Provide data directory', type=str)

# Define optional arguments
# parser.add_argument('--save_dir', help='Provide saving directory', type=str)
parser.add_argument('--arch', help='vgg13 can be used if this argument specified, otherwise vgg16 will be used',
                    type=str)
parser.add_argument('--learning_rate', help='Learning rate (default value 0.001)', type=float)
parser.add_argument('--hidden_units', help='Hidden units in Classifier (default value is 5120)', type=int)
parser.add_argument('--epochs', help='Number of epochs', type=int)
parser.add_argument('--gpu', help="Option to use GPU", type=str)

# Convert arguments to objects and assign to 'args'
args = parser.parse_args()


def load_data(data_dir='flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)

    print('Training, validation, and testing data loaded!')
    return train_data, validation_data, test_data, trainloader, validationloader, testloader


def load_label_mapping():
    with open('flower_to_name.json', 'r') as f:
        flower_to_name = json.load(f)

    print("Label to name mapping loaded!")
    return flower_to_name


def check_gpu(gpu):
    if not gpu:
        device = torch.device('cpu')
        print('Using CPU')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print('CUDA wasn\'t found on device, using CPU instead')
        print('Using GPU')
    return device


def train_network(gpu, arch='vgg16', hidden_units=5120, learning_rate=0.001, e=10):
    # Load pre-trained network
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    print('Pre-trained {0} model loaded!'.format(model.__class__.__name__))

    # Freeze feature parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
    print('Model parameters frozen!')

    # Create classifier
    classifier = nn.Sequential(OrderedDict([
        ('FC1', nn.Linear(25088, hidden_units)),
        ('ReLu1', nn.ReLU()),
        ('Dropout1', nn.Dropout(p=0.15)),

        ('FC2', nn.Linear(hidden_units, 512)),
        ('ReLu2', nn.ReLU()),
        ('Dropout2', nn.Dropout(p=0.15)),

        ('FC3', nn.Linear(512, 102)),
        ('Output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier  # Replace classifier with custom classifier
    print('New classifier created!')

    # See if CUDA is available
    device = check_gpu(gpu)

    # Define loss
    criterion = nn.NLLLoss()
    print('Negative Log Likelihood Loss defined!')

    # Define optimizer (only train classifier parameters)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print('Adam Optimizer defined!')

    # Send model to device
    model.to(device)
    print('Model sent to {0}!'.format(device))

    # Train network
    epochs = e  # Number of epochs to train for
    batch_number = 0  # Keep track of batch number
    print_every = 30  # Number of batches before printing again
    train_losses, validation_losses, validation_accuracies = [], [], []

    start = time()
    print('Training...')

    for epoch in range(epochs):  # Iterate over epochs
        running_loss = 0  # Keep track of training loss
        model.train()  # Set model to train mode (turn on dropout)

        for inputs, labels in trainloader:  # Training loop
            batch_number += 1

            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

            optimizer.zero_grad()  # Zero gradients

            logps = model.forward(inputs)  # Feed forward (log probabilities)
            loss = criterion(logps, labels)  # Calculate loss
            loss.backward()  # Back propagation
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Append current batch loss to current epoch loss

            # Print every ____ batches
            if batch_number % print_every == 0:
                validation_loss = 0  # Keep track of validation loss
                validation_accuracy = 0  # Accumulate accuracy across validation batches
                model.eval()  # Set model to evaluation mode (turn off dropout)

                with torch.no_grad():  # Don't compute gradients

                    for inputs, labels in validationloader:  # Validation loop

                        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
                        logps = model.forward(inputs)  # Feed forward (log probabilities)
                        batch_loss = criterion(logps, labels)  # Calculate loss

                        validation_loss += batch_loss.item()  # Append current batch loss to current epoch loss

                        # Calculate Test Accuracy
                        ps = torch.exp(logps)  # Calculate probabilities
                        top_p, top_class = ps.topk(1, dim=1)  # Get highest class in each probability
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Take note of training loss, validation loss, valdiation accuracy for current epoch/batch
                train_losses.append(running_loss / len(trainloader))
                validation_losses.append(validation_loss / len(validationloader))
                validation_accuracies.append(validation_accuracy / len(validationloader))

                # Print
                print("Epoch: {}/{} | ".format(epoch + 1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss / len(trainloader)),
                      "Validation Loss: {:.4f} | ".format(validation_loss / len(validationloader)),
                      "Validation Accuracy: {:.4f}".format(validation_accuracy / len(validationloader)))

    end = time()
    training_time = end - start
    print('Training Complete!')
    print("Training took {:0.4f} minutes~".format(training_time / 60))
    return train_losses, validation_losses, validation_accuracies, model, device, criterion, optimizer, epochs, training_time


def test_network(gpu):
    test_loss = 0
    test_accuracy = 0
    test_losses, test_accuracies = [], []

    # See if CUDA is available
    device = check_gpu(gpu)

    print('Testing network with testing set!')
    start = time()

    model.eval()  # Set model to evaluation mode (turn off dropout)
    with torch.no_grad():  # Don't compute gradients
        for inputs, labels in testloader:  # Testing loop
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            logps = model.forward(inputs)  # Feed forward (log probabilities)
            batch_loss = criterion(logps, labels)  # Calculate loss

            test_loss += batch_loss.item()  # Accumulate batch loss

            # accuracy
            ps = torch.exp(logps)  # Calculate probabilities
            top_p, top_class = ps.topk(1, dim=1)  # Get highest class in each probability
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            test_losses.append(test_loss / len(testloader))
            test_accuracies.append(test_accuracy / len(testloader))

            print("Test Lost: {:.4f} | ".format(test_loss / len(testloader)),
                  "Test Accuracy: {:.2f}".format((test_accuracy * 100 / len(testloader))) + "%")

    end = time()
    testing_time = end - start
    print('Testing Complete!')
    print("Testing took {:0.4f} seconds~".format(testing_time))
    return test_losses, test_accuracies, testing_time


def save_checkpoint(train_data, model):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'architecture': 'vgg16',
                  'model': model,
                  'classifier': model.classifier,
                  'criterion': criterion,
                  'optimizer': optimizer,
                  'epochs': epochs,
                  'learning_rate': 0.001,
                  'training_time_seconds': training_time,
                  'testing_time_seconds': testing_time,
                  'train_losses': train_losses,
                  'validation_losses': validation_losses,
                  'test_losses': test_losses,
                  'validation_accuracies': validation_accuracies,
                  'test_accuracies': test_accuracies,
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    print('Checkpoint made!')
    return model


train_data, validation_data, test_data, trainloader, validationloader, testloader = load_data(data_dir=args.data_dir)
flower_to_name = load_label_mapping()
train_losses, validation_losses, validation_accuracies, model, device, criterion, optimizer, epochs, training_time = \
    train_network(args.gpu, args.arch, args.hidden_units, args.learning_rate, args.epochs)
test_losses, test_accuracies, testing_time = test_network(args.gpu)
model = save_checkpoint(train_data, model)
