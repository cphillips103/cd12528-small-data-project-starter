#
# Starter code for Part 1 of the Small Data Solutions Project
#
#
# The Approach
# Load in pre-trained weights from a previous dataset
# Freeze all the weights in the lower layers
# Replace the 
# Train only the custom classifier and optimize
#


# Import pytorch libraries needed for working with data
import torch
import torch.nn as nn


# Used to wrap an iterable around the Dataset
from torch.utils.data import DataLoader
# Stores the samples and their corresponding labels
from torchvision import datasets

# Import torchvision
from torchvision.transforms import ToTensor
#from torchvision import transforms
from torchvision.transforms import v2
import torchvision.models as models



# For using GPU if available
from torch import optim, cuda
from torch.optim import lr_scheduler


from TrainModel import train_model
from TestModel import test_model


# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Whether to train on a gpu
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f'Training will be on: {device}')

# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


#
# Directory variables for images
# Loading from folder in same directory as the code
# Image folder includes three folders (train,val,test)
# Each subfolder contains various categories of images
# with Beach, Desert, Forest images
#
DATA_DIR = "imagedata-50/"
TRAIN_DATA_DIR = DATA_DIR+"train/"
VALID_DATA_DIR = DATA_DIR+"val/"
TEST_DATA_DIR = DATA_DIR+"test/"


#
# Begining of Data Setup
# Iterate through each category in image training directory
#

# Predefining batch size of 10
# i.e. each element in the dataloader iterable will return a batch
# of 10 features and labels.
batch_size = 10


# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomRotation(degrees=20),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        v2.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=5),
        v2.RandomGrayscale(p = 0.1),
        v2.RandomHorizontalFlip(p= 0.5),
        v2.CenterCrop(size = 64),  # Image net standards
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean,
                             std)  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    v2.Compose([
        v2.Resize(size = 256),
        v2.CenterCrop(size = 224),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean, std)
    ]),
    # Test does not use augmentation
    'test':
    v2.Compose([
        v2.Resize(size = 256),
        v2.CenterCrop(size = 224),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean, std)
    ]),
}

# Create a variable that contains the class_names. You can get them from the ImageFolder
# Datasets from each folder
data = {
    'train':
    datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=VALID_DATA_DIR, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=TEST_DATA_DIR, transform=image_transforms['test'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

train_iter = iter(dataloaders['train'])
features, labels = next(train_iter)
features.shape, labels.shape

number_of_classes = len(data['train'].classes)
print(f'There are {number_of_classes} different classes.')


number_of_targets = len(data['train'].targets)
print(f'There are {number_of_targets} different images.')

#
# End of Data Setup
#

#Set up Transforms (train, val, and test)
# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

def get_pretrained_model(model_name):

    if model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT')

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        number_of_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(number_of_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, number_of_classes), nn.LogSoftmax(dim=1))

    # Move to gpu if available
    if device:
        model = model.to('cuda')

    return model

model = get_pretrained_model('vgg16')

#
# Print Parameters
#

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')    


# Keep track of our mapping of class values to the indices
model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

# Convert class dictionary to list for TestModel
class_dict = {val: key for key, val in model.class_to_idx.items()}
class_names = list(class_dict.values())
print(f'Class names used in model: {", ".join(class_names)}')

#
# Train model with these hyperparameters
# 1. num_epochs - iterate through the train DataLoader. 1 Pass = Epoch
# 2. criterion  - loss (criterion): keeps track of the loss itself and the gradients of the loss with respect to the model parameters (weights)
# 3. optimizer  - Optimizer: updates the parameters (weights) with the gradients
# 4. train_lr_scheduler 
#

num_epochs=200
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# look at the parameters (weights) that will be updated by the optimizer during training.
for p in optimizer.param_groups[0]['params']:
    if p.requires_grad:
        print(p.shape)
        
train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

lam = lambda epoch: 0.85 ** epoch
train_lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lam)


# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
   trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
   test_model(test_loader, trained_model, class_names)

if __name__ == '__main__':
    main()
    print("done")