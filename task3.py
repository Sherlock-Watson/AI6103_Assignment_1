from preprocess import get_train_valid
from preprocess import train_by_params
import torch
import numpy as np
import torch.utils.data as data

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

    lr_list = [0.5, 0.05, 0.01]
    epochs = 15
    for learning_rate in lr_list:
        print(f"train for leaning rate {learning_rate} and no scheduler: ")
        train_by_params(train_loader, valid_loader, epochs, learning_rate)

if __name__ == '__main__':
    main()
