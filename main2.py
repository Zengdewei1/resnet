import torch
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from ResNet34 import ResNet34
from torch import optim

def train(model,data,target,loss_func,optimizer):
    optimizer.zero_grad()
    output=model(data)
    predictions = output.max(1,keepdim=True)[1]
    correct=predictions.eq(target.view_as(predictions)).sum().item()
    acc=correct / len(target)
    loss=loss_func(output,target)
    loss.backward()
    optimizer.step()
    return acc , loss

def test(model,test_loader,loss_func,use_cuda):
    acc_all=0
    loss_all=0
    step=0
    with torch.no_grad():
        for data ,target in test_loader:
            step += 1
            if use_cuda:
                data=data.cuda()
                target=target.cuda()
            output=model(data)
            predictions=output.max(1,keepdim=True)[1]
            correct=predictions.eq(target.view_as(predictions)).sum().item()
            acc=correct / len(target)
            loss=loss_func(output,target)
            acc_all += acc
            loss_all += loss
    return acc_all / step,loss_all / step
def main():
    dataset_type='cifar10'
    
    num_epochs=100
    batch_size=64
    eval_step=1000
    use_cuda=torch.cuda.is_available()
    
    dir_list = ('../data', '../data/MNIST', '../data/CIFAR-10')
    for directory in dir_list:
        if not os.path.exists(directory):
            os.mkdir(directory)
    if dataset_type=='mnist':
        train_loader=DataLoader(datasets.MNIST(root='../data/MNIST',train=True,download=True,transform=transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),batch_size=batch_size,shuffle=True)
        test_loader=DataLoader(datasets.MNIST(root='../data/MNIST',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),batch_size=batch_size)
    elif dataset_type=='cifar10':
        train_loader=DataLoader(datasets.CIFAR10(root='../data/CIFAR-10',train=True,download=True,transform=transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),batch_size=batch_size,shuffle=True)
        test_loader=DataLoader(datasets.CIFAR10(root='../data/CIFAR-10',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),batch_size=batch_size)
    else:
        raise ValueError('Wrong!')
        
    model = ResNet34()
    if use_cuda:
        model=model.cuda()
    ce_loss=torch.nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=1e-3)
    train_step=0
    for _ in range(num_epochs):
        for data, target in train_loader:
            train_step += 1
            if use_cuda:
                data=data.cuda()
                target=target.cuda()
            acc, loss=train(model,data,target,ce_loss,optimizer)
            if train_step % 100 ==0:
                print('Train set:Step:{},Loss:{:.4f},Accuracy:{:.2f}'.format(train_step,loss,acc))
            if train_step % eval_step==0:
                acc, loss=test(model,test_loader,ce_loss,use_cuda)
                print('\nTest set: Step: {}, Loss: {:.4f},Accuracy: {:.2f}\n'.format(train_step,loss,acc))
                
if __name__ == '__main__':
    main()