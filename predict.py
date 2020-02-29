# Uses a trained network to predict the class for an input image
import torch
import numpy as np
import seaborn as sns
import matplotlib as plt
from torchvision import transforms
import torch.utils.data
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser(description="Parser of prediction script")
parser.add_argument('image_dir', help='Provide path to image', type=str)
parser.add_argument('load_dir', help='Provide path to checkpoint', type=str)
parser.add_argument('--top_k', help='Top K most likely classes', type=int)
# parser.add_argument('--category_names', help='JSON file of mapping of categories to real names', type=str)
parser.add_argument('--gpu', help="Option to use GPU", type=str)
args = parser.parse_args()


def load_checkpoint(filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)

    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    print('Model loaded!')
    return model


def process_image(image='flowers/test/1/image_06743.jpg'):
    # Open image using PIL.Image package
    pil_image = Image.open(image)

    # Define preprocesses
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    return preprocess(pil_image)


def imshow(image='flowers/test/1/image_06743.jpg', ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


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


def load_label_mapping():
    with open('flower_to_name.json', 'r') as f:
        flower_to_name = json.load(f)

    print("Label to name mapping loaded!")
    return flower_to_name


def predict(image_path, model, tensor_image, gpu, flower_to_name, topk=5):
    # Preprocess image
    tensor_image = process_image(image_path)
    tensor_image = tensor_image.unsqueeze(0)  # Place a dimension of size 1 at position 0

    device = check_gpu(gpu)

    # Send model to device
    model = model.to(device)
    tensor_image = tensor_image.to(device)

    # Calculate class probabilities
    model.eval()
    with torch.no_grad():
        logps = model.forward(tensor_image)  # Feed forward (log probabilities)

    # Calculate probabilities
    ps = torch.exp(logps)

    # Get top 5 probabilities and indices of corresponding classes
    top_probabilities, top_classes_indices = torch.topk(ps, topk)

    # Convert tensors to list
    top_probabilities = top_probabilities.tolist()[0]
    top_classes_indices = top_classes_indices.tolist()[0]

    # Get labels from model.class_to_idx
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}  # Reverse class_to_idx
    top_classes = [idx_to_class[index] for index in top_classes_indices]  # Map index of class to actual class
    top_labels = [flower_to_name[label] for label in top_classes]  # Map actual class to flower name

    return top_probabilities, top_classes, top_labels


def check_sanity(path, model, gpu, flower_to_name):
    # Preprocess image
    image = process_image(path)

    # Calculate probabilities
    probabilities, classes, labels = predict(path, model, image, gpu, flower_to_name)

    # Set up plot
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)

    # Set up title
    flower_num = path.split('/')[2]
    title = flower_to_name[flower_num]

    # Plot/display flower
    imshow(image, ax, title)

    # Plot bar chart
    plt.subplot(2, 1, 2)
    sns.barplot(x=probabilities, y=labels, color=sns.color_palette()[0])
    plt.show()


model = load_checkpoint(args.load_dir)
tensor_image = process_image(args.image_dir)
flower_to_name = load_label_mapping()
probabilities, classes, labels = predict(args.image_dir, model, tensor_image, args.gpu, flower_to_name, args.top_k)
check_sanity(args.image_dir, model, args.gpu, flower_to_name)
