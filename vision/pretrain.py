import torchvision
import torch
import argparse
import os

def parse_args() -> dict :
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='choose which model to get trained', default='ResNet34')
    parser.add_argument('-d', '--dataset', type=str, help='choose which dataset to train on', default='CIFAR10')
    parser.add_argument('-s', '--seed', type=int, help='random seed to be set for model training', default=0)
    parser.add_argument('--download', help='it is required to download the dataset before training', type=bool, default=False)
    parser.add_argument('-b', '--batchsize', help='batch size', type=int, default=16)
    parser.add_argument('-e', '--epoch', help='num of epoches', type=int, default=200)
    return parser.parse_args()

def test(model, dataloader, loss_func, i, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print('='*100)
    print(f'Epoch {i} eval: loss = {test_loss:>8f}, acc = {correct*100:>.2f}%')

def train(model, train_dataloader, test_dataloader, loss_func, optimizer, scheduler, device, epoch=200):
    model = model.to(device)
    for i in range(epoch):
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            loss = loss_func(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 200 == 0:
                print('Epoch ', i, ', step ', batch, ': loss = ', loss.item(), sep='')
        scheduler.step()
        test(model, test_dataloader, loss_func, i, device)
    return model

def main():
    # import pdb
    # pdb.set_trace()
    arg = parse_args()
    torch.random.manual_seed(arg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # choose dataset
    if arg.dataset.upper() == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10('./download/CIFAR-10/train', train=True, download=arg.download, transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.CIFAR10('./download/CIFAR-10/test', train=False, download=arg.download, transform=torchvision.transforms.ToTensor())
        num_classes=10
    elif arg.dataset.upper() == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100('./download/CIFAR-100/train', train=True, download=arg.download, transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.CIFAR100('./download/CIFAR-100/test', train=False, download=arg.download, transform=torchvision.transforms.ToTensor())
        num_classes=100
    else:
        raise ValueError('Unsupported dataset!')
    print("Dataset prepared")
    
    # choose model
    if arg.model == 'ResNet34':
        from torchvision.models.resnet import resnet34, ResNet
        model = resnet34(num_classes=num_classes)
    elif arg.model == 'MobileNetV3':
        from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNetV3
        model = mobilenet_v3_small(num_classes=num_classes)
    elif arg.model == 'EfficientNetV2':
        from torchvision.models.efficientnet import EfficientNet, efficientnet_v2_s
        model = efficientnet_v2_s(num_classes=num_classes)
    else:
        raise ValueError('Unsupported model!')
    print("Model prepared")
    
    # start training
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batchsize, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=arg.batchsize, shuffle=False)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    model = train(model, train_dataloader, test_dataloader, loss_func, optimizer, scheduler, device, epoch=arg.epoch)
    
    # save trained model
    output_dir = os.path.join('train', arg.model, arg.dataset, str(arg.seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, 'train.pth'))
        
if __name__ == '__main__':
    main()