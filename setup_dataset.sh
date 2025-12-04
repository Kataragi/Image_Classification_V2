#!/bin/bash
# Setup script for creating dataset directory structure

echo "🎨 Setting up Art Style Classification dataset structure..."

# Create main dataset directory
mkdir -p dataset

# Create 8 class directories
classes=("anime" "brush" "thick" "watercolor" "photo" "3dcg" "comic" "pixelart")

for class in "${classes[@]}"; do
    mkdir -p "dataset/$class"
    echo "✓ Created dataset/$class/"
done

# Create output directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p visualizations
mkdir -p results

echo ""
echo "✅ Dataset structure created successfully!"
echo ""
echo "📁 Directory structure:"
echo "   dataset/"
for class in "${classes[@]}"; do
    echo "   ├── $class/"
done
echo ""
echo "📋 Next steps:"
echo "   1. Place your images in the respective class folders"
echo "   2. Supported formats: .jpg, .jpeg, .png, .bmp, .webp"
echo "   3. Run: python train.py --dataset dataset"
echo ""
