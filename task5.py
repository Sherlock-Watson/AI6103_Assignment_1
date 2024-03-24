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


def get_transforms(dataset_dir: str):
    trainset = torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True)

    print(f"trainset size: {len(trainset)}")

    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    # mean = [0.0, 0.0, 0.0]
    # std = [0.0, 0.0, 0.0]

    img = np.array(trainset[0][0])
    img_h = img.shape[0]
    img_w = img.shape[1]
    print(f"imgh: {img_h}; imgw: {img_w}")

    pixel_count = img_h * img_w * len(trainset)

    for train_item, _ in trainset:
        train_item = np.array(train_item)
        mean += np.sum(train_item, axis=0)
        std += np.sum(train_item ** 2, axis=0)

    # get mean and deviation
    mean = [m / pixel_count for m in mean]
    std = [math.sqrt((s / pixel_count) - m ** 2) for s, m in zip(std, mean)]
    print('Mean of each color channel:', mean)
    print('Standard deviation of each color channel:', std)

    # get the new transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform


def get_train_valid(dataset_dir: str, seed: int):
    train_transform = get_transforms(dataset_dir)

    # generate trainset and valset
    trainset = torchvision.datasets.CIFAR100(dataset_dir, train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size],
                                                     generator=torch.Generator().manual_seed(seed))
    print(f"new_trainset size: {len(trainset)}")
    print(f"new_valset size: {len(valset)}")

    return trainset, valset


def train_by_params(train_loader, valid_loader, epochs, learning_rate, weight_decay=0, lr_scheduler=False):
    # model
    model = MobileNet(100)
    print(model)
    model.cuda()

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)

    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    for epoch in range(epochs):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        # training
        model.train()
        for img, labels in train_loader:
            img = img.cuda()
            labels = labels.cuda()

            batch_size = img.shape[0]
            optimizer.zero_grad()
            logits = model.forward(img)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            _, top_class = logits.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            training_loss += batch_size * loss.item()
            training_samples += batch_size
        # validation
        model.eval()
        for val_imgs, val_labels in valid_loader:
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
            loss = criterion(val_logits, val_labels.cuda())
            _, top_class = val_logits.topk(1, dim=1)
            equals = top_class == val_labels.cuda().view(*top_class.shape)
            val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
        assert val_samples == 10000
        # update stats
        stat_training_loss.append(training_loss / training_samples)
        stat_val_loss.append(val_loss / val_samples)
        stat_training_acc.append(training_acc / training_samples)
        stat_val_acc.append(val_acc / val_samples)
        # print
        print(
            f"Epoch {(epoch + 1):d}/{epochs:d}.. Learning rate: {learning_rate}.. Train loss: {(training_loss / training_samples):.4f}.. Train acc: {(training_acc / training_samples):.4f}.. Val loss: {(val_loss / val_samples):.4f}.. Val acc: {(val_acc / val_samples):.4f}")
        # lr scheduler
        if lr_scheduler:
            scheduler.step()
    # plot
    # plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, f"task3_lr_{learning_rate}.png")


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
    learning_rate = 0.01 
    weight_decay = 5e-4
    train_by_params(train_loader, valid_loader, epochs, learning_rate, weight_decay, True)
    print("train with best learning rate and weight decay 1e-4 and lr schedule")
    weight_decay = 1e-4
    train_by_params(train_loader, valid_loader, epochs, learning_rate, weight_decay, True)

if __name__ == '__main__':
    main()
