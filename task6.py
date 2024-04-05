from torchvision.transforms import v2
from preprocess import get_train_valid
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from mobilenet import MobileNet
import matplotlib.pyplot as plt

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_with_mixup(train_loader, valid_loader, test_loader, epochs, learning_rate, weight_decay=0, lr_scheduler=False):
    mixup = v2.MixUp(alpha=0.2, num_classes=100)
    # model
    model = MobileNet(100)
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
        for imgs, labels in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            mixed_x, y_a, y_b, lam = mixup_data(imgs, labels)
            batch_size = mixed_x.shape[0]
            optimizer.zero_grad()
            logits = model.forward(mixed_x)
            loss = mixup_criterion(logits, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            _, top_class = logits.topk(1, dim=1)
            equals = top_class == labels.cuda().view(*top_class.shape)
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
    print(f'stat_training_loss = {stat_training_loss}')
    print(f'stat_val_loss = {stat_val_loss}')
    print(f'stat_training_acc = {stat_training_acc}')
    print(f'stat_val_acc = {stat_val_acc}')
    test_loss = 0
    test_acc = 0
    test_samples = 0
    for test_imgs, test_labels in test_loader:
        batch_size = test_imgs.shape[0]
        test_logits = model.forward(test_imgs.cuda())
        test_loss = criterion(test_logits, test_labels.cuda())
        _, top_class = test_logits.topk(1, dim=1)
        equals = top_class == test_labels.cuda().view(*top_class.shape)
        test_acc += torch.sum(equals.type(torch.FloatTensor)).item()
        test_loss += batch_size * test_loss.item()
        test_samples += batch_size
    assert test_samples == 10000
    print('Test loss: ', test_loss / test_samples)
    print('Test acc: ', test_acc / test_samples)

def get_test_set(data_dir):
    mean = [0.5070751736356132, 0.4865488876838237, 0.44091785745237955]
    std = [0.26875306, 0.25662399, 0.27623447]
    # get the new transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=32, padding=4),
        transforms.Normalize(mean, std)
    ])
    testset = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
    return testset

if __name__== '__main__':
    # fix random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # train val test
    # AI6103 students: You need to create the dataloaders yourself
    trainset, valset = get_train_valid('datasets', 0)
    testset = get_test_set('datasets')

    batch_size = 128

    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    learning_rate = 0.05
    epochs = 300
    print("task 6")
    weight_decay = 5e-4
    train_with_mixup(train_loader, valid_loader, test_loader, epochs, learning_rate, weight_decay, True)
    