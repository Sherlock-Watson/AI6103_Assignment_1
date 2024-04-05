import torch
import torchvision
import torchvision.transforms as transforms
from mobilenet import MobileNet
import matplotlib.pyplot as plt

def get_transforms(dataset_dir: str):
    mean = [0.5070751736356132, 0.4865488876838237, 0.44091785745237955]
    std = [0.26875306, 0.25662399, 0.27623447]
    # get the new transform
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=32, padding=4),
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
    # get the class proportion
    train_class_count = [0] * 100
    for _, target in trainset:
        train_class_count[target] += 1
    print("class proportion: ")
    print(train_class_count)
    plt.figure(figsize=(27, 10))
    plt.xlabel('target')
    plt.ylabel('sum')
    plt.bar(range(1, 101), train_class_count)
    plt.savefig("class_distribution.png")

    return trainset, valset


def train_by_params(train_loader, valid_loader, epochs, learning_rate, weight_decay=0, lr_scheduler=False):
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
    # equals = torch.tensor([0.0])
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
        scheduler.step()
    print(f'stat_training_loss = {stat_training_loss}')
    print(f'stat_val_loss = {stat_val_loss}')
    print(f'stat_training_acc = {stat_training_acc}')
    print(f'stat_val_acc = {stat_val_acc}')
