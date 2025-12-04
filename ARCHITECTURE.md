# 🏗️ システムアーキテクチャ

## 📁 プロジェクト構造

```
Image_Classification_V2/
├── train.py                 # メイントレーニングスクリプト
├── inference.py             # 推論・可視化スクリプト
├── verify_setup.py          # 環境確認スクリプト
├── setup_dataset.sh         # データセットセットアップスクリプト
│
├── models/                  # モデル定義
│   ├── __init__.py
│   └── msa_net.py          # MSA-Net実装
│
├── utils/                   # ユーティリティ関数
│   └── __init__.py
│
├── dataset/                 # データセット (ユーザーが配置)
│   ├── anime/
│   ├── brush/
│   ├── thick/
│   ├── watercolor/
│   ├── photo/
│   ├── 3dcg/
│   ├── comic/
│   └── pixelart/
│
├── checkpoints/             # 学習済みモデル
│   ├── best_model.pth
│   ├── checkpoint_epoch_10.pth
│   └── final_model.pth
│
├── logs/                    # TensorBoardログ
│
├── visualizations/          # 可視化結果
│   ├── style_space.png
│   └── cluster_centers.json
│
├── requirements.txt         # Python依存関係
├── README.md               # メインドキュメント
├── QUICKSTART.md           # クイックスタートガイド
├── DATASET_EXAMPLE.md      # データセット例
└── ARCHITECTURE.md         # このファイル
```

## 🧠 モデルアーキテクチャ

### Base Model: ConvNeXtV2-Base-22k-384

```
Input Image (384x384)
    ↓
ConvNeXtV2 Backbone
    ├── Stem (Conv + Norm)
    ├── Stage 1 (128 channels)
    ├── Stage 2 (256 channels)
    ├── Stage 3 (512 channels)
    └── Stage 4 (1024 channels)
    ↓
Global Average Pooling
    ↓
Classifier Head
    ↓
Output (8 classes)
```

### MSA-Net Enhancement (Optional)

```
ConvNeXtV2 Features (1024 channels)
    ↓
Multi-Scale Attention Module
    ├── Channel Attention
    │   ├── Avg Pool → FC → ReLU → FC
    │   └── Max Pool → FC → ReLU → FC
    │   └── Sigmoid
    │
    └── Spatial Attention
        ├── Channel-wise Avg + Max
        └── Conv 7x7 → Sigmoid
    ↓
Style Aggregation Module
    ├── Texture Branch (Conv 3x3)
    └── Color Branch (Conv 1x1)
    ↓
Fusion (Conv 1x1)
    ↓
Global Pooling
    ↓
Enhanced Classifier
    ├── Dropout (0.3)
    ├── Linear (1024 → 512)
    ├── ReLU
    ├── Dropout (0.2)
    └── Linear (512 → 8)
    ↓
Output (8 classes)
```

## 🔄 トレーニングフロー

### 1. データロード

```python
Dataset
    ├── クラスフォルダ読み込み
    ├── 画像パス収集
    ├── クラス別カウント
    └── クラス重み計算
    ↓
Train/Val Split (85/15)
    ├── Virtual Split (物理分割なし)
    └── DataLoader作成
```

### 2. 段階的解像度学習

```
Resolution: 384x384 (初期)
    ↓
Val Lossモニタリング
    ↓
3 epoch連続で改善 < 0.02?
    ↓ Yes
Resolution: 512x512
    ↓
Val Lossモニタリング
    ↓
3 epoch連続で改善 < 0.02?
    ↓ Yes
Resolution: 768x768
    ↓
Val Lossモニタリング
    ↓
3 epoch連続で改善 < 0.02?
    ↓ Yes
Resolution: 1024x1024 (最終)
```

### 3. トレーニングループ

```python
for epoch in epochs:
    # Training
    for batch in train_loader:
        images, labels → GPU (double-check)
        ↓
        forward pass
        ↓
        loss calculation (weighted)
        ↓
        backward pass
        ↓
        optimizer step
        ↓
        update progress bar

    # Validation
    for batch in val_loader:
        images, labels → GPU (double-check)
        ↓
        forward pass (no grad)
        ↓
        collect predictions
        ↓
        calculate metrics

    # Logging
    ├── TensorBoard (loss, acc, confusion matrix)
    ├── Best model save
    ├── Periodic checkpoint
    └── Early stopping check

    # Resolution update check
    └── should_increase_resolution()
```

## 🎯 推論フロー

### 単一画像推論

```python
Image Path
    ↓
Load & Preprocess
    ├── RGB変換
    ├── Resize (model resolution)
    ├── Normalize
    └── To Tensor → GPU
    ↓
Model Forward
    ↓
Softmax
    ↓
Results
    ├── Predicted Class
    ├── Confidence Score
    ├── Top-3 Predictions
    └── All Probabilities
```

### フォルダ一括推論

```python
Folder Path
    ↓
Collect Image Paths
    ↓
For each image:
    ├── Predict
    └── Store result
    ↓
Aggregate Results
    ├── Class Distribution
    └── Statistics
    ↓
Save JSON
```

### スタイル空間可視化

```python
Training Dataset
    ↓
Feature Extraction
    ├── Forward to last layer
    └── Collect features (N x 1024)
    ↓
PCA (1024 → 50)
    ↓
t-SNE (50 → 2)
    ↓
Cluster Center Calculation
    ├── Per-class mean in 2D space
    └── Save absolute coordinates
    ↓
Test Images (optional)
    ├── Extract features
    ├── Apply same PCA + t-SNE
    └── Plot in same space
    ↓
Visualization
    ├── Scatter plot (train data)
    ├── Star markers (cluster centers)
    ├── X markers (test data)
    └── Save PNG
```

## 🔧 主要コンポーネント

### 1. ArtStyleDataset

```python
class ArtStyleDataset(Dataset):
    - root_dir: データセットパス
    - transform: データ拡張
    - classes: クラスリスト
    - class_to_idx: クラス→インデックス
    - samples: (画像パス, ラベル)リスト
    - class_counts: クラス別枚数
```

### 2. MSANet

```python
class MSANet(nn.Module):
    - base_model: ConvNeXtV2
    - style_aggregation: スタイル集約モジュール
    - classifier: 拡張分類器

    def extract_features(): 可視化用特徴抽出
    def forward(): 分類
```

### 3. TrainLoop

```python
def train_epoch():
    - Forward pass
    - Loss calculation
    - Backward pass
    - Optimizer step
    - Progress update

def validate():
    - Forward pass (no grad)
    - Metrics calculation
    - Confusion matrix
    - TensorBoard logging
```

## 📊 データフロー

### GPU転送ダブルチェック

```python
# Training
images = images.to(device)
labels = labels.to(device)
assert images.is_cuda  # ✓ Check
assert labels.is_cuda  # ✓ Check

# Model
model = model.to(device)
assert next(model.parameters()).is_cuda  # ✓ Check
```

### クラス重み付け

```python
total_samples = N
class_counts = [n1, n2, ..., n8]

weights[i] = total_samples / (num_classes * class_counts[i])

# 例: anime=1000枚, pixelart=100枚
weight_anime = 5000 / (8 * 1000) = 0.625
weight_pixelart = 5000 / (8 * 100) = 6.25
# → pixelartの重要度が10倍
```

## 🎨 可視化パイプライン

### 次元削減

```
High-dimensional features (1024D)
    ↓
PCA (1024D → 50D)
    - Preserve major variance
    - Speed up t-SNE
    ↓
t-SNE (50D → 2D)
    - Non-linear projection
    - Preserve local structure
    - Perplexity: 30
    ↓
2D coordinates
    - X: dimension 1
    - Y: dimension 2
```

### クラスタリング

```python
For each class:
    class_points = embeddings[labels == class_id]
    center = mean(class_points)
    save to cluster_centers.json

# 将来の推論で再利用
test_embedding → compare with centers → classify
```

## ⚙️ ハイパーパラメータ

### デフォルト設定

| パラメータ | デフォルト | 説明 |
|----------|----------|------|
| 初期解像度 | 384 | 最初の学習解像度 |
| 最終解像度 | 1024 | 段階的に上昇 |
| バッチサイズ | 16 | GPU VRAMに応じて調整 |
| 学習率 | 1e-4 | AdamW optimizer |
| Weight Decay | 0.05 | 正則化 |
| Val Split | 0.15 | 15%をバリデーション |
| Resolution Threshold | 0.02 | 解像度上昇の閾値 |

### データ拡張

**Training:**
- Resize
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation ±0.2)
- Normalize (ImageNet stats)

**Validation/Inference:**
- Resize
- Normalize

## 🔒 安全機能

### Early Stopping

```python
if args.early_stop_on_increase:
    if val_loss > previous_val_loss:
        save_checkpoint()
        break
```

### チェックポイント保存

- **Best Model**: Val Loss最小時
- **Periodic**: N epoch毎
- **Early Stop**: Val Loss上昇時
- **Final**: 学習終了時

各チェックポイント内容:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
    'val_acc': val_acc,
    'resolution': current_resolution,
    'classes': class_names,
    'use_msa_net': bool
}
```

## 🚀 最適化

### メモリ効率

- Virtual dataset split (物理コピーなし)
- Pin memory for faster GPU transfer
- Gradient checkpointing (オプション)
- Mixed precision training (実装可能)

### 計算効率

- Multi-worker data loading (num_workers=4)
- Batch processing
- Progressive resolution (低解像度から開始)
- Cosine annealing scheduler

## 📈 拡張可能性

### 追加クラス

```python
# dataset/に新しいフォルダを追加するだけ
dataset/
└── new_style/  # 新しい画風
    └── images...
```

### カスタムモデル

```python
# models/custom_model.py
class CustomModel(nn.Module):
    ...

# train.pyで使用
model = CustomModel(num_classes)
```

### カスタム損失関数

```python
# Focal Lossなど
criterion = FocalLoss(weight=class_weights)
```

---

## 📚 参考文献

- ConvNeXtV2: https://arxiv.org/abs/2301.00808
- timm library: https://github.com/huggingface/pytorch-image-models
- PyTorch: https://pytorch.org/
