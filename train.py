#!/usr/bin/env python3
"""
Art Style Classification Training Script
Supports progressive resolution training with ConvNeXtV2 and optional MSA-Net
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import MSA-Net module
from models.msa_net import MSANet


class ArtStyleDataset(Dataset):
    """Dataset for art style classification"""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect all image paths
        self.samples = []
        self.class_counts = {cls: 0 for cls in self.classes}

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    self.class_counts[class_name] += 1

        print(f"\n📊 Dataset Statistics:")
        for cls_name, count in self.class_counts.items():
            print(f"  {cls_name}: {count} images")
        print(f"  Total: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (384, 384), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def calculate_class_weights(dataset: ArtStyleDataset) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets"""
    total_samples = len(dataset)
    num_classes = len(dataset.classes)

    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        class_counts[label] += 1

    # Calculate weights: inverse frequency
    weights = total_samples / (num_classes * class_counts)

    print(f"\n⚖️  Class Weights (for imbalanced dataset):")
    for i, cls_name in enumerate(dataset.classes):
        print(f"  {cls_name}: {weights[i]:.4f}")

    return weights


def get_transforms(resolution: int, is_train: bool = True):
    """Get data transforms for given resolution"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_model(num_classes: int, use_msa_net: bool = False, device: str = 'cuda'):
    """Create ConvNeXtV2 model with optional MSA-Net"""

    print(f"\n🏗️  Creating model...")
    print(f"  Base: ConvNeXtV2-Base-22k-384")
    print(f"  MSA-Net: {'Enabled' if use_msa_net else 'Disabled'}")

    # Load ConvNeXtV2 base model
    try:
        base_model = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            pretrained=True,
            num_classes=num_classes
        )
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ⚠ Error loading model: {e}")
        print(f"  Attempting to download...")
        base_model = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            pretrained=True,
            num_classes=num_classes
        )

    if use_msa_net:
        # Wrap with MSA-Net
        model = MSANet(base_model, num_classes)
    else:
        model = base_model

    # Move to device and double-check
    model = model.to(device)

    # Double-check GPU transfer
    if device == 'cuda':
        assert next(model.parameters()).is_cuda, "❌ Model not on CUDA!"
        print(f"  ✓ Model successfully transferred to GPU")

    return model


def plot_confusion_matrix(cm, class_names, epoch, writer):
    """Plot confusion matrix to TensorBoard"""
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - Epoch {epoch}')

    writer.add_figure('Confusion Matrix', fig, epoch)
    plt.close(fig)


def validate(model, val_loader, criterion, device, epoch, writer, class_names):
    """Validation loop with confusion matrix"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            # Double-check GPU transfer
            images = images.to(device)
            labels = labels.to(device)

            if device == 'cuda':
                assert images.is_cuda, "❌ Validation images not on CUDA!"
                assert labels.is_cuda, "❌ Validation labels not on CUDA!"

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, epoch, writer)

    return val_loss, val_acc


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, pbar):
    """Single training epoch"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Double-check GPU transfer
        images = images.to(device)
        labels = labels.to(device)

        if device == 'cuda':
            assert images.is_cuda, "❌ Training images not on CUDA!"
            assert labels.is_cuda, "❌ Training labels not on CUDA!"

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar (single line)
        pbar.set_postfix({
            'loss': f'{train_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        pbar.update(1)

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


def should_increase_resolution(val_loss_history: List[float], threshold: float = 0.02) -> bool:
    """Check if resolution should be increased (3 consecutive epochs with improvement < threshold)"""
    if len(val_loss_history) < 4:
        return False

    recent_losses = val_loss_history[-4:]
    improvements = [recent_losses[i] - recent_losses[i+1] for i in range(3)]

    return all(imp < threshold for imp in improvements)


def main():
    parser = argparse.ArgumentParser(description='Art Style Classification Training')

    # Dataset args
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')

    # Model args
    parser.add_argument('--use-msa-net', action='store_true',
                        help='Use MSA-Net (Multimodal Style Aggregation Network)')

    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--resolution-threshold', type=float, default=0.02,
                        help='Val loss improvement threshold for resolution increase')

    # Save args
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early-stop-on-increase', action='store_true',
                        help='Stop and save if val loss increases from previous epoch')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')

    # Device args
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  Running on CPU")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Initialize TensorBoard
    writer = SummaryWriter('logs')

    # Load full dataset with initial resolution
    print(f"\n📂 Loading dataset from: {args.dataset}")

    current_resolution = 384
    resolutions = [384, 512, 768, 1024]
    resolution_idx = 0

    full_dataset = ArtStyleDataset(
        args.dataset,
        transform=get_transforms(current_resolution, is_train=True)
    )

    # Calculate class weights
    class_weights = calculate_class_weights(full_dataset).to(device)

    # Split dataset (virtual split, not physical)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\n📊 Dataset Split:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    num_classes = len(full_dataset.classes)
    model = create_model(num_classes, args.use_msa_net, device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')
    val_loss_history = []

    print(f"\n🎯 Starting training...")
    print(f"  Initial resolution: {current_resolution}x{current_resolution}")
    print(f"  Progressive resolutions: {resolutions}")

    for epoch in range(1, args.epochs + 1):
        # Create progress bar for this epoch
        pbar = tqdm(
            total=len(train_loader),
            desc=f'Epoch {epoch}/{args.epochs} [Res: {current_resolution}]',
            leave=True,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, pbar
        )

        pbar.close()

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer, full_dataset.classes
        )

        val_loss_history.append(val_loss)

        # Logging
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Resolution', current_resolution, epoch)

        # Early stopping on val loss increase
        if args.early_stop_on_increase and epoch > 1:
            if val_loss > val_loss_history[-2]:
                print(f"\n⚠️  Val loss increased: {val_loss_history[-2]:.4f} → {val_loss:.4f}")
                print(f"💾 Saving checkpoint and stopping training...")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'resolution': current_resolution,
                    'classes': full_dataset.classes,
                    'use_msa_net': args.use_msa_net
                }, os.path.join(args.output_dir, f'early_stop_epoch_{epoch}.pth'))

                break

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'resolution': current_resolution,
                'classes': full_dataset.classes,
                'use_msa_net': args.use_msa_net
            }, os.path.join(args.output_dir, 'best_model.pth'))

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'resolution': current_resolution,
                'classes': full_dataset.classes,
                'use_msa_net': args.use_msa_net
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Check if should increase resolution
        if should_increase_resolution(val_loss_history, args.resolution_threshold):
            if resolution_idx < len(resolutions) - 1:
                resolution_idx += 1
                current_resolution = resolutions[resolution_idx]

                print(f"\n📈 Increasing resolution: {resolutions[resolution_idx-1]} → {current_resolution}")

                # Update datasets with new resolution
                full_dataset.transform = get_transforms(current_resolution, is_train=True)
                val_dataset.dataset.transform = get_transforms(current_resolution, is_train=False)

                # Recreate data loaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )

                # Reset val loss history for new resolution
                val_loss_history = []

        # Step scheduler
        scheduler.step()

    print(f"\n✅ Training completed!")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Final Resolution: {current_resolution}x{current_resolution}")

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'resolution': current_resolution,
        'classes': full_dataset.classes,
        'use_msa_net': args.use_msa_net
    }, os.path.join(args.output_dir, 'final_model.pth'))

    writer.close()


if __name__ == '__main__':
    main()
