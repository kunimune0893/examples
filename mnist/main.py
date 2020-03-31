from __future__ import print_function

import random
import argparse
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

EN_CROSS_ENTROPY = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if EN_CROSS_ENTROPY:
            output = x
        else:
            output = F.log_softmax(x, dim=1)
        return output

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, nllloss=False):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.cls = classes
        self.nll = nllloss
    
    def forward(self, pred, target):
        if not self.nll:
            pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #smooth_label = smooth_one_hot(target, output.size(-1), args.smooth)
        #loss = criterion(output, smooth_label)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0) # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    return correct / len(train_loader.dataset), train_loss / len(train_loader.dataset)

def test(args, model, device, test_loader, criterion, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for ii, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if writer is not None and ii == 0:
                print("output.shape=", output.shape, "pred.shape=", pred.shape, "target.shape=", target.shape )
                #print("pred=", pred)
                #print("target=", target)
                grid = torchvision.utils.make_grid(data)
                writer.add_image('images', grid, 0)
                writer.add_graph(model, data)
    
    test_loss /= len(test_loader.dataset)
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct / len(test_loader.dataset), test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--smooth', type=float, default=0.0, metavar='SM',
                        help='for smooth labeling (default: 0.0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    print( "2020/03/20 23:35" )
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print( "args.seed=", args.seed, "device=", device )
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    model = Net().to(device)
    
    if EN_CROSS_ENTROPY:
        criterion_trn = LabelSmoothingLoss(len(train_loader.dataset.classes), args.smooth)
        criterion_tst = nn.NLLLoss(reduction='sum')
    else:
        criterion_trn = nn.NLLLoss()
        criterion_tst = nn.NLLLoss(reduction='sum')
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    writer = SummaryWriter()
    
    for epoch in range(1, args.epochs + 1):
        trn_acc1, trn_loss = train(args, model, device, train_loader, criterion_trn, optimizer, epoch)
        val_acc1, val_loss = test(args, model, device, test_loader, criterion_tst)
        
        writer.add_scalar('Loss/train', trn_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/train', trn_acc1, epoch)
        writer.add_scalar('Accuracy/test', val_acc1, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    writer.close()

if __name__ == '__main__':
    main()
