from preprocess import get_train_valid
from preprocess import train_by_params
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from mobilenet import MobileNet
import random
from utils import plot_loss_acc
import matplotlib.pyplot as plt
import math

def main():
    # fix random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # train val test
    # AI6103 students: You need to create the dataloaders yourself
    trainset, valset = get_train_valid('datasets', 0)

    batch_size = 128

    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=1)


    epochs = 300
    learning_rate = 0.05 
    weight_decay = 5e-4
    train_by_params(train_loader, valid_loader, epochs, learning_rate, weight_decay, True)
    print("train with best learning rate and weight decay 1e-4 and lr schedule")
    weight_decay = 1e-4
    train_by_params(train_loader, valid_loader, epochs, learning_rate, weight_decay, True)

if __name__ == '__main__':
    main()
