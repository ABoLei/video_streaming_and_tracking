import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import json
from model import create_model

class ClassificationDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        
        # Load image and convert to RGB or grayscale based on transform
        try:
            image = Image.open(file_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image

def get_file_paths_and_labels(root_dir, label_map):
    """Get file paths and labels for a given root directory"""
    file_paths = []
    gt_sports = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    for sport in sorted(os.listdir(root_dir)):
        sport_root = os.path.join(root_dir, sport)
        if os.path.isdir(sport_root):
            for file_name in sorted(os.listdir(sport_root)):
                file_path = os.path.join(sport_root, file_name)
                if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in valid_extensions):
                    file_paths.append(file_path)
                    gt_sports.append(sport)
    
    labels = [label_map[sport] for sport in gt_sports]
    return file_paths, labels

def create_data_transforms(input_size=224, use_rgb=True):
    """Create data transforms for training and validation"""
    
    if use_rgb:
        # RGB transforms
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Grayscale transforms
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    return train_transform, val_transform

@torch.no_grad()
def batch_topk_correct(logits, labels, k=5):
    """Calculate top-k accuracy for a batch"""
    topk = torch.topk(logits, k, dim=1).indices
    return topk.eq(labels.view(-1, 1)).any(dim=1).float().sum().item()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    train_loss, t1_correct, t5_correct, n = 0.0, 0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Train {epoch+1:02d}/{total_epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        n += bs
        train_loss += loss.item() * bs
        t1_correct += (outputs.argmax(1) == labels).sum().item()
        t5_correct += batch_topk_correct(outputs, labels, k=5)

        pbar.set_postfix(
            loss=f"{train_loss/max(n,1):.4f}",
            top1=f"{(t1_correct/max(n,1))*100:.2f}%",
            top5=f"{(t5_correct/max(n,1))*100:.2f}%"
        )

    return train_loss / max(n, 1), t1_correct / max(n, 1) * 100, t5_correct / max(n, 1) * 100

@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss, v1_correct, v5_correct, vn = 0.0, 0, 0, 0
    
    pbar = tqdm(val_loader, desc="Valid")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        vn += bs
        val_loss += loss.item() * bs
        v1_correct += (outputs.argmax(1) == labels).sum().item()
        v5_correct += batch_topk_correct(outputs, labels, k=5)

        pbar.set_postfix(
            loss=f"{val_loss/max(vn,1):.4f}",
            top1=f"{(v1_correct/max(vn,1))*100:.2f}%",
            top5=f"{(v5_correct/max(vn,1))*100:.2f}%"
        )

    return val_loss / max(vn, 1), v1_correct / max(vn, 1) * 100, v5_correct / max(vn, 1) * 100

def train():
    parser = argparse.ArgumentParser(description='Train Sports Classification Model')
    parser.add_argument('--data_dir', type=str, default='dataset/train', help='Training data directory')
    parser.add_argument('--model_type', type=str, default='efficient', choices=['efficient', 'lightweight'], help='Model type')
    parser.add_argument('--use_rgb', action='store_true', help='Use RGB input (default: grayscale)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--student_id', type=str, default='313551121', help='Student ID')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save model')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create label mapping
    sports = sorted(os.listdir(args.data_dir))
    label_map = {sport: i for i, sport in enumerate(sports)}
    num_classes = len(sports)
    print(f"Number of classes: {num_classes}")
    
    # Get file paths and labels
    train_file_paths, train_labels = get_file_paths_and_labels(args.data_dir, label_map)
    print(f"Training samples: {len(train_file_paths)}")
    
    # Split training data into train and validation (80-20 split)
    np.random.seed(42)
    indices = np.random.permutation(len(train_file_paths))
    split_idx = int(0.8 * len(train_file_paths))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_paths = [train_file_paths[i] for i in train_indices]
    train_lbls = [train_labels[i] for i in train_indices]
    val_paths = [train_file_paths[i] for i in val_indices]
    val_lbls = [train_labels[i] for i in val_indices]
    
    print(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
    
    # Create transforms
    input_channels = 3 if args.use_rgb else 1
    train_transform, val_transform = create_data_transforms(use_rgb=args.use_rgb)
    
    # Create datasets and loaders
    train_dataset = ClassificationDataset(train_paths, train_lbls, transform=train_transform)
    val_dataset = ClassificationDataset(val_paths, val_lbls, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    # Create model
    model = create_model(args.model_type, num_classes, input_channels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_accuracy = 0.0
    model_save_path = os.path.join(args.save_dir, f"w_{args.student_id}.pth")
    history = {'train_loss': [], 'train_top1': [], 'train_top5': [], 
               'val_loss': [], 'val_top1': [], 'val_top5': []}
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_top1, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # Validate
        val_loss, val_top1, val_top5 = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_top1'].append(train_top1)
        history['train_top5'].append(train_top5)
        history['val_loss'].append(val_loss)
        history['val_top1'].append(val_top1)
        history['val_top5'].append(val_top5)
        
        # Print epoch results
        print(f"\nEpoch [{epoch+1:02d}/{args.epochs}]")
        print(f"Train: loss {train_loss:.4f}, top1 {train_top1:.2f}%, top5 {train_top5:.2f}%")
        print(f"Valid: loss {val_loss:.4f}, top1 {val_top1:.2f}%, top5 {val_top5:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on top-5 accuracy
        if val_top5 > best_accuracy:
            best_accuracy = val_top5
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'model_config': {
                    'model_type': args.model_type,
                    'num_classes': num_classes,
                    'input_channels': input_channels,
                    'total_params': total_params
                }
            }, model_save_path)
            print(f"New best model saved! Top-5 accuracy: {best_accuracy:.2f}%")
    
    # Save training history
    history_path = os.path.join(args.save_dir, f"history_{args.student_id}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation Top-5 accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print(f"History saved to: {history_path}")

if __name__ == "__main__":
    train()
