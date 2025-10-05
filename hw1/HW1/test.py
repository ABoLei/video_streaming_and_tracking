import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from model import create_model

class TestDataset(Dataset):
    def __init__(self, root_dir, class_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get image paths
        valid_extensions = {".jpg", ".jpeg", ".png"}
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in sorted(os.listdir(root_dir))
            if os.path.splitext(fname)[1].lower() in valid_extensions
        ]
        
        # Get class names from training directory
        self.classes = sorted([
            d for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return dummy label

def create_test_transform(input_size=224, use_rgb=True):
    """Create test transform"""
    if use_rgb:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    return transform

def test_with_tta(model, image, device, num_tta=5):
    """Test Time Augmentation for better predictions"""
    model.eval()
    predictions = []
    
    # Original prediction
    with torch.no_grad():
        pred = model(image.to(device))
        predictions.append(F.softmax(pred, dim=1))
    
    # TTA predictions with slight variations
    tta_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1),
    ]
    
    for i in range(min(num_tta-1, len(tta_transforms))):
        # Apply TTA transform
        tta_image = tta_transforms[i](transforms.ToPILImage()(image.squeeze(0)))
        tta_tensor = transforms.ToTensor()(tta_image).unsqueeze(0)
        
        # Normalize based on input channels
        if image.shape[1] == 1:  # Grayscale
            tta_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tta_tensor)
        else:  # RGB
            tta_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tta_tensor)
        
        with torch.no_grad():
            pred = model(tta_tensor.to(device))
            predictions.append(F.softmax(pred, dim=1))
    
    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred

def test():
    parser = argparse.ArgumentParser(description='Test Sports Classification Model')
    parser.add_argument('--test_dir', type=str, default='dataset/test', help='Test data directory')
    parser.add_argument('--train_dir', type=str, default='dataset/train', help='Training data directory (for class names)')
    parser.add_argument('--weight_path', type=str, help='Path to model weights')
    parser.add_argument('--model_type', type=str, default='efficient', choices=['efficient', 'lightweight'], help='Model type')
    parser.add_argument('--use_rgb', action='store_true', help='Use RGB input (default: grayscale)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--student_id', type=str, default='313551121', help='Student ID')
    parser.add_argument('--output_csv', type=str, help='Output CSV file path')
    parser.add_argument('--use_tta', action='store_true', help='Use Test Time Augmentation')
    
    args = parser.parse_args()
    
    # Auto-detect parameters if weight_path is provided
    if args.weight_path and os.path.exists(args.weight_path):
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            args.model_type = config.get('model_type', args.model_type)
            input_channels = config.get('input_channels', 1)
            args.use_rgb = (input_channels == 3)
            num_classes = config.get('num_classes', 100)
            print(f"Loaded model config: {config}")
        else:
            input_channels = 1 if not args.use_rgb else 3
            num_classes = 100
    else:
        # Use default weight path
        if not args.weight_path:
            args.weight_path = f"w_{args.student_id}.pth"
        input_channels = 1 if not args.use_rgb else 3
        num_classes = 100
    
    # Set output CSV path
    if not args.output_csv:
        args.output_csv = f"pred_{args.student_id}.csv"
    
    print(f"Using weights: {args.weight_path}")
    print(f"Model type: {args.model_type}")
    print(f"Input channels: {input_channels} ({'RGB' if args.use_rgb else 'Grayscale'})")
    print(f"Output CSV: {args.output_csv}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test transform
    transform = create_test_transform(use_rgb=args.use_rgb)
    
    # Create test dataset
    dataset = TestDataset(root_dir=args.test_dir, class_dir=args.train_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Test samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # Create model
    model = create_model(args.model_type, num_classes, input_channels)
    
    # Load weights
    if os.path.exists(args.weight_path):
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle potential module prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded successfully!")
    else:
        print(f"Warning: Weight file {args.weight_path} not found. Using random weights.")
    
    model = model.to(device).eval()
    
    # Generate predictions
    results = []
    idx_start = 0
    
    print("Generating predictions...")
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Testing"):
            if args.use_tta:
                # Use TTA for each image individually
                batch_predictions = []
                for i in range(images.size(0)):
                    single_image = images[i:i+1]
                    pred = test_with_tta(model, single_image, device)
                    batch_predictions.append(pred)
                logits = torch.cat(batch_predictions, dim=0)
            else:
                # Standard inference
                images = images.to(device, non_blocking=True)
                logits = model(images)
                logits = F.softmax(logits, dim=1)
            
            # Get top-5 predictions
            top5 = torch.topk(logits, k=min(5, num_classes), dim=1).indices.cpu()
            
            # Get corresponding file paths
            batch_paths = [dataset.image_paths[i] for i in range(idx_start, idx_start + images.size(0))]
            idx_start += images.size(0)
            
            # Process each prediction
            for j, path in enumerate(batch_paths):
                fname = os.path.basename(path)
                idxs = top5[j].tolist()
                labels = [dataset.classes[i] for i in idxs]
                
                # Ensure we have exactly 5 predictions
                while len(labels) < 5:
                    labels.append(labels[-1] if labels else dataset.classes[0])
                
                results.append([fname] + labels[:5])
    
    # Create DataFrame and save
    df = pd.DataFrame(results, columns=["file_name", "pred1", "pred2", "pred3", "pred4", "pred5"])
    
    # Sort by filename to ensure consistent ordering
    df = df.sort_values('file_name').reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.', exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    
    print(f"Predictions saved to: {args.output_csv}")
    print(f"Generated predictions for {len(results)} images")
    
    # Display first few predictions
    print("\nFirst 5 predictions:")
    print(df.head())

if __name__ == "__main__":
    test()
