from torch.optim.lr_scheduler import StepLR
from model import resnet34, resnet56
from dataset import get_cifar_loaders

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def train_one_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    # sum of loss over all the samples
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )

    train_loader, val_loader, test_loader = get_cifar_loaders(dataset_name=args.dataset,
                                                  batch_size=args.batch_size)

    if args.arch == 'resnet34':
        model = resnet34(num_classes=args.num_classes)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(args.save_dir, exist_ok=True)

    save_path = ""
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, test_loader, criterion)

        scheduler.step()

        print(f'Epoch [{epoch}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            })

        # save model in case of best val acc
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f'best_{args.arch}_{args.dataset}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved to {save_path}')
            if args.wandb:
                wandb.save(save_path)

    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    test_loss, test_acc = validate(model, device, test_loader, criterion)
    print(f'Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    if args.wandb:
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet on with wandb logging')
    parser.add_argument('--arch', default='resnet34', type=str, choices=['resnet18','resnet34','resnet56'],
                        help='ResNet architecture')
    parser.add_argument('--dataset', default='CIFAR100', type=str, choices=['CIFAR10','CIFAR100'],
                        help='Dataset to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--num_classes', default=100, type=int,
                        help='Number of classes')
    parser.add_argument('--epochs', default=70, type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=int,
                        help='Momentum value')
    parser.add_argument('--step_size', default=30, type=int,
                        help='Scheduler step size')
    parser.add_argument('--gamma', default=0.1, type=int,
                        help='Scheduler gamma')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--save_dir', default='./checkpoints', type=str,
                        help='Directory to save models')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (user or team name)")
    parser.add_argument("--wandb_project", type=str, default="decoder-only-transformer",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional run name for Weights & Biases")
    parser.add_argument('--project', default='resnet-cifar', type=str,
                        help='wandb project name')

    args = parser.parse_args()
    main(args)