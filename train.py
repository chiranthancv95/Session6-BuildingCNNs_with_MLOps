import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Print epoch training accuracy
    train_accuracy = 100. * correct / total
    print(f'Training Accuracy: {correct}/{total} ({train_accuracy:.2f}%)')
    return train_accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Training settings
    batch_size = 64
    epochs = 20
    lr = 0.01
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Enhanced Data transformations with more aggressive augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation((-7, 7)),
        transforms.RandomAffine(degrees=0, translate=(0.7, 0.7), scale=(0.85, 1.15), fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    #transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets with separate transforms
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, 
                           pin_memory=True)

    model = MNISTNet().to(device)
    
    count_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {count_parameters}")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, 
                         weight_decay=2e-4)
    
    # Modified learning rate schedule
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.25,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch: {epoch}')
        train_accuracy = train(model, device, train_loader, optimizer, epoch)
        test_accuracy = test(model, device, test_loader)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, 'mnist_model.pth')
            print(f'New best test accuracy: {test_accuracy:.2f}%')
        
        print(f'Current LR: {scheduler.get_last_lr()[0]:.6f}')
        scheduler.step()

if __name__ == '__main__':
    main() 