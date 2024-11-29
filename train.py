import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    # Training settings
    batch_size = 32
    epochs = 20
    lr = 0.05
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Enhanced Data transformations with more augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation((-10.0, 10.0), fill=(0,)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(0,)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
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
                         weight_decay=3e-4)
    
    # Modified learning rate schedule
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Longer warm-up
        div_factor=20,  # Less aggressive initial lr reduction
        final_div_factor=1e3,  # Less aggressive final lr reduction
        anneal_strategy='cos'  # Cosine annealing
    )

    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, 'mnist_model.pth')
            print(f'New best accuracy: {accuracy:.2f}%')
        
        print(f'Current LR: {scheduler.get_last_lr()[0]:.6f}')
        scheduler.step()

if __name__ == '__main__':
    main() 