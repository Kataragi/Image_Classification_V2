#!/usr/bin/env python3
"""
Art Style Classification Inference Script
Supports single image inference, batch folder inference, and style space visualization
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

# Import MSA-Net module
from models.msa_net import MSANet


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""

    print(f"\n📦 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model info
    num_classes = len(checkpoint['classes'])
    use_msa_net = checkpoint.get('use_msa_net', False)
    resolution = checkpoint.get('resolution', 384)

    print(f"  Classes: {checkpoint['classes']}")
    print(f"  MSA-Net: {'Enabled' if use_msa_net else 'Disabled'}")
    print(f"  Resolution: {resolution}x{resolution}")

    # Create base model
    base_model = timm.create_model(
        'convnextv2_base.fcmae_ft_in22k_in1k_384',
        pretrained=False,
        num_classes=num_classes
    )

    # Wrap with MSA-Net if needed
    if use_msa_net:
        model = MSANet(base_model, num_classes)
    else:
        model = base_model

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Double-check GPU transfer
    if device == 'cuda':
        assert next(model.parameters()).is_cuda, "❌ Model not on CUDA!"
        print(f"  ✓ Model successfully loaded on GPU")

    return model, checkpoint['classes'], resolution, use_msa_net


def get_transform(resolution: int):
    """Get inference transform"""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(image_path: str, model, classes: List[str], resolution: int, device: str):
    """Predict art style for a single image"""

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

    transform = get_transform(resolution)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Double-check GPU transfer
    if device == 'cuda':
        assert image_tensor.is_cuda, "❌ Image tensor not on CUDA!"

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()

    # Get top 3 predictions
    top3_prob, top3_idx = probabilities.topk(3, dim=1)
    top3_predictions = [
        (classes[idx.item()], prob.item())
        for idx, prob in zip(top3_idx[0], top3_prob[0])
    ]

    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'top3': top3_predictions,
        'probabilities': probabilities.cpu().numpy()[0]
    }


def predict_folder(folder_path: str, model, classes: List[str], resolution: int, device: str):
    """Predict art styles for all images in a folder"""

    folder = Path(folder_path)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Collect image paths
    image_paths = [
        p for p in folder.iterdir()
        if p.suffix.lower() in valid_extensions
    ]

    if not image_paths:
        print(f"❌ No images found in {folder_path}")
        return []

    print(f"\n📁 Processing {len(image_paths)} images...")

    results = []

    for img_path in tqdm(image_paths, desc="Inference"):
        result = predict_single_image(img_path, model, classes, resolution, device)

        if result:
            result['image_path'] = str(img_path)
            results.append(result)

    return results


def extract_features(model, data_loader, device: str, use_msa_net: bool):
    """Extract features for visualization"""

    features_list = []
    labels_list = []

    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)

            # Double-check GPU transfer
            if device == 'cuda':
                assert images.is_cuda, "❌ Images not on CUDA!"

            # Extract features
            if use_msa_net:
                features = model.extract_features(images)
            else:
                # For base model, use features before classifier
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(images)
                    if features.dim() == 4:
                        features = features.mean(dim=[2, 3])  # Global average pooling
                else:
                    # Fallback: use output of penultimate layer
                    features = model(images)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


def visualize_style_space(
    model,
    train_loader,
    test_images: List[str],
    classes: List[str],
    resolution: int,
    device: str,
    use_msa_net: bool,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    output_path: str = 'visualizations/style_space.png'
):
    """Visualize style space using t-SNE or PCA"""

    print("\n🎨 Visualizing style space...")

    # Extract features from training data
    print("  Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device, use_msa_net)

    # Extract features from test images if provided
    test_features = None
    test_predictions = []

    if test_images:
        print(f"  Extracting test features from {len(test_images)} images...")
        test_data = []

        transform = get_transform(resolution)

        for img_path in test_images:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                test_data.append(image_tensor)
            except Exception as e:
                print(f"  ⚠ Error loading {img_path}: {e}")

        if test_data:
            test_batch = torch.stack(test_data).to(device)

            # Double-check GPU transfer
            if device == 'cuda':
                assert test_batch.is_cuda, "❌ Test batch not on CUDA!"

            with torch.no_grad():
                if use_msa_net:
                    test_features = model.extract_features(test_batch).cpu().numpy()
                else:
                    if hasattr(model, 'forward_features'):
                        feats = model.forward_features(test_batch)
                        if feats.dim() == 4:
                            feats = feats.mean(dim=[2, 3])
                        test_features = feats.cpu().numpy()
                    else:
                        test_features = model(test_batch).cpu().numpy()

                # Get predictions
                outputs = model(test_batch)
                _, predicted = outputs.max(1)
                test_predictions = predicted.cpu().numpy()

    # Combine train and test features for consistent embedding
    if test_features is not None:
        all_features = np.concatenate([train_features, test_features], axis=0)
    else:
        all_features = train_features

    # Apply dimensionality reduction (t-SNE or PCA)
    print("  Applying t-SNE dimensionality reduction...")

    # Use PCA first to reduce dimensions if features are high-dimensional
    if all_features.shape[1] > 50:
        pca = PCA(n_components=50)
        all_features = pca.fit_transform(all_features)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(all_features)

    # Split embeddings back
    train_embeddings = embeddings_2d[:len(train_features)]

    if test_features is not None:
        test_embeddings = embeddings_2d[len(train_features):]
    else:
        test_embeddings = None

    # Calculate cluster centers for each class (absolute coordinates)
    cluster_centers = {}
    for i, class_name in enumerate(classes):
        class_mask = train_labels == i
        if class_mask.any():
            center = train_embeddings[class_mask].mean(axis=0)
            cluster_centers[class_name] = center

    # Save cluster centers for future reference
    cluster_centers_path = 'visualizations/cluster_centers.json'
    os.makedirs('visualizations', exist_ok=True)

    with open(cluster_centers_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in cluster_centers.items()}, f, indent=2)

    print(f"  💾 Cluster centers saved to: {cluster_centers_path}")

    # Create visualization
    plt.figure(figsize=(14, 10))

    # Plot training data
    colors = sns.color_palette('husl', len(classes))

    for i, class_name in enumerate(classes):
        class_mask = train_labels == i
        if class_mask.any():
            plt.scatter(
                train_embeddings[class_mask, 0],
                train_embeddings[class_mask, 1],
                c=[colors[i]],
                label=f'{class_name} (train)',
                alpha=0.6,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )

            # Plot cluster center
            center = cluster_centers[class_name]
            plt.scatter(
                center[0], center[1],
                c=[colors[i]],
                marker='*',
                s=500,
                edgecolors='black',
                linewidth=2
            )

    # Plot test data
    if test_embeddings is not None:
        for i, (emb, pred_label) in enumerate(zip(test_embeddings, test_predictions)):
            plt.scatter(
                emb[0], emb[1],
                c=[colors[pred_label]],
                marker='X',
                s=200,
                edgecolors='black',
                linewidth=2,
                label=f'Test: {classes[pred_label]}' if i == 0 else ''
            )

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Art Style Space Visualization', fontsize=16, fontweight='bold')

    # Set axis ranges if provided
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  💾 Visualization saved to: {output_path}")

    plt.close()

    return cluster_centers


def main():
    parser = argparse.ArgumentParser(description='Art Style Classification Inference')

    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # Inference mode args
    parser.add_argument('--image', type=str,
                        help='Path to single image for inference')
    parser.add_argument('--folder', type=str,
                        help='Path to folder containing images for batch inference')

    # Visualization args
    parser.add_argument('--visualize', action='store_true',
                        help='Create style space visualization')
    parser.add_argument('--train-dataset', type=str,
                        help='Path to training dataset (required for visualization)')
    parser.add_argument('--test-images', type=str, nargs='+',
                        help='Test images to plot in style space')
    parser.add_argument('--x-range', type=float, nargs=2,
                        help='X-axis range for visualization (min max)')
    parser.add_argument('--y-range', type=float, nargs=2,
                        help='Y-axis range for visualization (min max)')
    parser.add_argument('--output-viz', type=str, default='visualizations/style_space.png',
                        help='Output path for visualization')

    # Output args
    parser.add_argument('--output-json', type=str,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")

    # Load model
    model, classes, resolution, use_msa_net = load_model(args.checkpoint, device)

    # Single image inference
    if args.image:
        print(f"\n🖼️  Single Image Inference")
        print(f"   Image: {args.image}")

        result = predict_single_image(args.image, model, classes, resolution, device)

        if result:
            print(f"\n✨ Prediction Result:")
            print(f"   Predicted Class: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"\n   Top 3 Predictions:")
            for i, (cls, prob) in enumerate(result['top3'], 1):
                print(f"     {i}. {cls}: {prob:.4f}")

            if args.output_json:
                os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
                with open(args.output_json, 'w') as f:
                    # Convert numpy types to native Python types
                    result_serializable = {
                        'predicted_class': result['predicted_class'],
                        'confidence': float(result['confidence']),
                        'top3': [(cls, float(prob)) for cls, prob in result['top3']],
                        'probabilities': result['probabilities'].tolist()
                    }
                    json.dump(result_serializable, f, indent=2)
                print(f"\n💾 Results saved to: {args.output_json}")

    # Folder batch inference
    elif args.folder:
        print(f"\n📁 Folder Batch Inference")
        print(f"   Folder: {args.folder}")

        results = predict_folder(args.folder, model, classes, resolution, device)

        if results:
            print(f"\n✨ Inference Results ({len(results)} images):")

            # Count predictions by class
            class_counts = {cls: 0 for cls in classes}
            for result in results:
                class_counts[result['predicted_class']] += 1

            print(f"\n   Distribution:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     {cls}: {count} images ({100*count/len(results):.1f}%)")

            if args.output_json:
                os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
                with open(args.output_json, 'w') as f:
                    # Convert numpy types to native Python types
                    results_serializable = []
                    for result in results:
                        results_serializable.append({
                            'image_path': result['image_path'],
                            'predicted_class': result['predicted_class'],
                            'confidence': float(result['confidence']),
                            'top3': [(cls, float(prob)) for cls, prob in result['top3']],
                            'probabilities': result['probabilities'].tolist()
                        })
                    json.dump(results_serializable, f, indent=2)
                print(f"\n💾 Results saved to: {args.output_json}")

    # Style space visualization
    elif args.visualize:
        if not args.train_dataset:
            print("❌ Error: --train-dataset is required for visualization")
            return

        print(f"\n🎨 Style Space Visualization")

        # Load training dataset
        from torch.utils.data import DataLoader
        from train import ArtStyleDataset

        train_dataset = ArtStyleDataset(
            args.train_dataset,
            transform=get_transform(resolution)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        # Parse axis ranges
        x_range = tuple(args.x_range) if args.x_range else None
        y_range = tuple(args.y_range) if args.y_range else None

        # Create visualization
        visualize_style_space(
            model,
            train_loader,
            args.test_images or [],
            classes,
            resolution,
            device,
            use_msa_net,
            x_range,
            y_range,
            args.output_viz
        )

        print("\n✅ Visualization complete!")

    else:
        print("❌ Error: Please specify --image, --folder, or --visualize")
        parser.print_help()


if __name__ == '__main__':
    main()
