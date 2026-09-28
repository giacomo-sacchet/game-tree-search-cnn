# Connect to drive
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

# Copy dataset folder from drive into the virtual machine
source = '/content/drive/MyDrive/thesis/SportCNN/Dataset_sport.zip'
dest   = '/content/Dataset_sport.zip'
!cp $source $dest

# Unzip the dataset
# Crea la cartella ed estrae il contenuto lì dentro
!mkdir -p /content/Dataset_sport
!unzip -q /content/Dataset_sport.zip -d /content/Dataset_sport
# Verifica che ora esistano le sottocartelle delle classi
!ls /content/Dataset_sport

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.nn.functional as F
import os
import pandas as pd
from google.colab import drive
import zipfile
from torchvision.datasets import ImageFolder

# M1.1
# Nuovi valori di normalizzazione (standard per immagini naturali)
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

# M7.0
# Training: Random variations to improve generalization
T_train = transforms.Compose([
    transforms.Resize(72),                # Resize slightly larger first
    transforms.RandomCrop(64, padding=4), # Randomly crop to 64x64
    transforms.RandomHorizontalFlip(),    # 50% chance to flip left-to-right
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
# M7.0

T_valid = transforms.Compose([
    transforms.Resize(72),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
# M1.1

############################################################

# A custom dataset class to handle subsets with specific transformations
class SubDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, indices, transform=None):
        # Initalize dataset variables
        self.base_dataset = base_dataset
        self.indices      = indices
        self.transform    = transform

    def __len__(self):
        # Return the number of items in the subset
        return len(self.indices)

    def __getitem__(self, i):
        # Retrieves the image and label based on the original dataset index
        x, y = self.base_dataset[self.indices[i]]
        # Apply transformation (like normalization) if provided
        if self.transform:
            x = self.transform(x)
        return x, y

def LoadDS(dataset, transform_train, transform_valid, split=0.8) :
    # Determine the number of training samples based on the split ratio
    nb = len(dataset)
    n_train = int(split * nb)

    # Use a fix seed for reproducible random shuffling of indices
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(nb, generator=g).tolist()

    # Slice the shuffled indices for training and validation sets
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]

    # Create SubDataset instances for training (80%) and validation (20%)
    train_ds = SubDataset(dataset, train_idx, transform_train)
    val_ds   = SubDataset(dataset, val_idx,   transform_valid)

    print("Dataset LOADED")
    return train_ds, val_ds

#########################################################

# Building block for the CNN: convolution -> relu activation -> max pooling
class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            # First convolutional layer
            nn.Conv2d(c_in, c_out, 3, padding=1),  # 3x3 convolution maintaining spatial dimensions
            nn.BatchNorm2d(c_out),                 # M5.0 Batch normalization for stabilizing training
            nn.ReLU(inplace=True),                 # Non linear activation function

            # Second convolutional layer
            nn.Conv2d(c_out, c_out, 3, padding=1),  
            nn.BatchNorm2d(c_out),                 
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)                        # Downsample by factor 2
        )

# Main CNN architecture for CIFAR-10 classification
class CifarCNNBase(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # Feature extractor reduces spatial dimentions: L -> L/2 -> L/4 -> L/8
        self.features = nn.Sequential(
            ConvBlock(3, 64),     #  Input: 3 channels, Output: 64 x L/2 x L/2
            ConvBlock(64, 128),   # Input: 64, Output: 128 x L/4 x L/4
            ConvBlock(128, 256),  # Input: 128, Output: 256 x L/8 x L/8
        )

        # Classifier section to convert features into class probabilities
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduce each feature map to a single 1x1 value (256 x 1 x 1
            nn.Flatten(),                  # Flatten to a 1D vector of size 256
            nn.Linear(256, num_classes)    # Final linear layer for 10 classes
        )

    def forward(self, x):
        # Passes input through features then the classifier
        x = self.features(x)
        x = self.classifier(x)
        return x

################################################################

class Scenario : pass

def createScenario(lr):
    # Helper function to initialize training parameters and data loaders
    S = Scenario()
    S.log_folder = "/content/logs" # Use Linux-style path
    if not os.path.exists(S.log_folder):
        os.makedirs(S.log_folder)

    # Create a unique csv filename based on the learning rate
    S.CSVname    = os.path.join(S.log_folder, "LR_" + format(lr, ".0e") +".csv")

    # train/val
    S.epochs      = 40
    S.batch_size  = 32

    # Initialize DataLoaders for batching and shuffling
    S.train_batch = DataLoader(train_ds, batch_size=S.batch_size, shuffle=True)
    S.valid_batch = DataLoader(val_ds,   batch_size=S.batch_size, shuffle=False)

    # Hardware acceleration check (GPU if available)
    S.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(S.device)

    # Model, loss function and optimizer setup
    S.model     = CifarCNNBase(num_classes=8)
    S.loss_fn   = nn.CrossEntropyLoss()
    S.lr = lr

    # M4.0
    # Switched from Adam to SGD. Added momentum=0.9 for better convergence
    S.optimizer = torch.optim.SGD(S.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4 )
    # M4.0

    # M6.0
    # Define the scheduler: drop LR at epoch 20 and epoch 35
    # This results in: 0.01 -> 0.001 -> 0.0001
    S.scheduler = torch.optim.lr_scheduler.MultiStepLR(S.optimizer, milestones=[20, 35], gamma=0.1)
    #M 6.0

    return S

#######################################################################

from torch.utils.tensorboard import SummaryWriter

def train(S,name):
    # List to store metrics for each epoch to export to CSV later
    info = [ ]

    S.model = S.model.to(S.device)
    for epoch in range(S.epochs):

        # --- TRAINING PHASE ---
        train_GlobLoss = train_accuracy = 0
        S.model.train() # Set model to training mode
        for x, y in S.train_batch:
            x = x.to(S.device)
            y = y.to(S.device)

            # Reset gradients, forward pass, calculate loss and bacckpropagate
            S.optimizer.zero_grad()
            logits = S.model(x)
            loss = S.loss_fn(logits,y)
            loss.backward()
            train_GlobLoss += loss.item() * x.size(0)
            S.optimizer.step()

            # Calculate batch training accuracy
            predictions     = logits.argmax(dim=1)
            train_accuracy += (predictions == y).sum().item()

        # --- VALIDATION PHASE ---
        val_conf = valid_GlobLoss = valid_accuracy = 0
        S.model.eval()  # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation for efficiency
            for x, y in S.valid_batch:
                x = x.to(S.device)
                y = y.to(S.device)
                logits = S.model(x)
                loss = S.loss_fn(logits, y)
                valid_GlobLoss += loss.item() * x.size(0)

                # Accuracy
                predictions     = logits.argmax(dim=1)
                valid_accuracy += (predictions == y).sum().item()

                # Calculate confidence (max probability) for validation samples
                probs = F.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
                val_conf += conf.sum().item()

        S.scheduler.step()  # Updates the LR based on the milestone

        # Normalize metrics by dataset size
        nbtrain, nbvalid = len(S.train_batch.dataset), len(S.valid_batch.dataset)
        train_GlobLoss  /= nbtrain
        train_accuracy  /= nbtrain
        valid_GlobLoss  /= nbvalid
        valid_accuracy  /= nbvalid
        val_conf        /= nbvalid

        # Log progress to console
        print( f"{epoch+1}/{S.epochs} - "
            f"tLoss {train_GlobLoss:.3f}  - vLoss {valid_GlobLoss:.3f}  -  "
            f"tAcc  {train_accuracy:.3f}  -  vAcc {valid_accuracy:.3f}  -  cAcc {val_conf:.3f}")

        # Store metrics for CSV output
        info.append([epoch+1,train_accuracy,valid_accuracy,train_GlobLoss,valid_GlobLoss,val_conf])

    # Convert logged info into a DataFrame and save to CSV
    columns = [ "epoch", "acc/train", "acc/valid", "loss/train", "loss/valid", "conf/val" ]
    df = pd.DataFrame(info, columns=columns)
    df.to_csv( S.CSVname , index = False )

#####################################################

# --- MAIN EXECUTION ---
# M1.2
# Punta alla cartella dove abbiamo estratto i dati
data_path = '/content/Dataset_sport/Dataset'

# Uso di ImageFolder per gestire il dataset con 8 classi
dataset = ImageFolder(root=data_path, transform=None)

# Suddivisione in train e validation (80/20 come nel codice originale)
train_ds, val_ds = LoadDS(dataset, T_train, T_valid)

# Use the standard lr (1e-2) for SGD optimizer
lr_best = 1e-2
S = createScenario(lr = lr_best)
train(S, "__test_run")
# M1.2