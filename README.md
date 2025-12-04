# 🎨 Art Style Classification V2

高度な画風分類システム。ConvNeXtV2とオプションのMSA-Net (Multimodal Style Aggregation Network)を使用した8クラスの画風分類を実現。

## 📋 対応画風クラス

1. **アニメ塗り** (anime) - アニメ調のデジタルペイント
2. **ブラシ塗り** (brush) - ブラシストロークが特徴的な塗り
3. **厚塗り** (thick) - 油絵のような厚みのある塗り
4. **水彩** (watercolor) - 水彩画風
5. **写真** (photo) - 実写・フォトリアリスティック
6. **3DCG** (3dcg) - 3Dレンダリング
7. **白黒マンガ** (comic) - モノクロコミック
8. **ピクセルアート** (pixelart) - ドット絵

## 🚀 主要機能

### トレーニング (train.py)
- ✅ 段階的解像度学習 (384→512→768→1024)
- ✅ 不均衡データセット対応の自動クラス重み付け
- ✅ 自動バリデーション分割 (デフォルト15%)
- ✅ MSA-Net統合 (オプション)
- ✅ TensorBoard可視化 (Confusion Matrix含む)
- ✅ tqdmプログレスバー
- ✅ 柔軟なチェックポイント保存
- ✅ Early Stopping機能
- ✅ GPU転送ダブルチェック

### 推論 (inference.py)
- ✅ 単一画像推論
- ✅ フォルダ一括推論
- ✅ スタイル空間可視化 (t-SNE)
- ✅ Top-3予測
- ✅ 信頼度スコア
- ✅ JSON出力

---

## 🛠️ 環境構築 (WSL2 + CUDA 12.8 + PyTorch)

### 前提条件
- Windows 11 with WSL2
- NVIDIA GPU (CUDA対応)
- NVIDIA Driver (最新版推奨)

### Step 1: WSL2のセットアップ

```bash
# Windows PowerShell (管理者権限)で実行
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

再起動後、Ubuntu 22.04を起動。

### Step 2: CUDA 12.8のインストール

```bash
# システムアップデート
sudo apt update && sudo apt upgrade -y

# 必要なパッケージ
sudo apt install -y build-essential

# CUDA Toolkit 12.8のインストール
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

# 環境変数設定
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# CUDA確認
nvcc --version
nvidia-smi
```

### Step 3: Pythonとvenv環境

```bash
# Python 3.10以上をインストール
sudo apt install -y python3.10 python3.10-venv python3-pip

# プロジェクトディレクトリ作成
mkdir -p ~/projects
cd ~/projects
git clone <your-repo-url> Image_Classification_V2
cd Image_Classification_V2

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate
```

### Step 4: PyTorchとその他のパッケージ

```bash
# PyTorch (CUDA 12.1対応版) インストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# その他の依存関係
pip install -r requirements.txt

# GPU確認
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

正常にインストールされていれば、以下のような出力が得られます:
```
CUDA Available: True
CUDA Version: 12.1
Device: NVIDIA GeForce RTX 4090
```

---

## 📁 データセット構造

```
dataset/
├── anime/
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
├── brush/
│   ├── image001.jpg
│   └── ...
├── thick/
│   └── ...
├── watercolor/
│   └── ...
├── photo/
│   └── ...
├── 3dcg/
│   └── ...
├── comic/
│   └── ...
└── pixelart/
    └── ...
```

**注意**:
- 各クラスフォルダに画像を配置するだけでOK
- サポート形式: JPG, PNG, BMP, WebP
- 枚数が不均衡でも自動で重み付け調整されます

---

## 🎓 トレーニング

### 基本的なトレーニング

```bash
python train.py \
  --dataset dataset \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4
```

### MSA-Netを使用したトレーニング

```bash
python train.py \
  --dataset dataset \
  --use-msa-net \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4
```

### 詳細設定

```bash
python train.py \
  --dataset dataset \
  --val-split 0.2 \
  --use-msa-net \
  --epochs 150 \
  --batch-size 32 \
  --lr 5e-5 \
  --resolution-threshold 0.01 \
  --save-every 5 \
  --early-stop-on-increase \
  --output-dir checkpoints
```

### オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--dataset` | データセットパス | `dataset` |
| `--val-split` | バリデーション分割比率 | `0.15` |
| `--use-msa-net` | MSA-Netを使用 | `False` |
| `--epochs` | エポック数 | `100` |
| `--batch-size` | バッチサイズ | `16` |
| `--lr` | 学習率 | `1e-4` |
| `--resolution-threshold` | 解像度上昇の閾値 | `0.02` |
| `--save-every` | 保存間隔(エポック) | `10` |
| `--early-stop-on-increase` | Val Loss上昇で停止 | `False` |
| `--output-dir` | 出力ディレクトリ | `checkpoints` |

### TensorBoard起動

```bash
tensorboard --logdir logs
```

ブラウザで `http://localhost:6006` にアクセス。

---

## 🔮 推論

### 単一画像推論

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image test_image.jpg
```

**出力例:**
```
✨ Prediction Result:
   Predicted Class: anime
   Confidence: 0.9234

   Top 3 Predictions:
     1. anime: 0.9234
     2. brush: 0.0512
     3. thick: 0.0154
```

### フォルダ一括推論

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --folder test_images/ \
  --output-json results/predictions.json
```

**出力例:**
```
📁 Processing 150 images...
Inference: 100%|████████████| 150/150

✨ Inference Results (150 images):

   Distribution:
     anime: 65 images (43.3%)
     watercolor: 32 images (21.3%)
     brush: 28 images (18.7%)
     thick: 15 images (10.0%)
     photo: 10 images (6.7%)

💾 Results saved to: results/predictions.json
```

### スタイル空間可視化

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --test-images test1.jpg test2.jpg test3.jpg \
  --output-viz visualizations/style_space.png
```

カスタム軸範囲:
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --test-images test1.jpg test2.jpg \
  --x-range -50 50 \
  --y-range -40 40 \
  --output-viz visualizations/style_space_custom.png
```

---

## 📊 段階的解像度学習について

トレーニングは以下のように自動的に解像度を上げていきます:

1. **384x384** - 初期学習
2. **512x512** - Val Lossが3エポック連続で0.02以下の改善
3. **768x768** - さらに頭打ちしたら
4. **1024x1024** - 最終段階

この仕組みにより:
- 初期段階では高速に全体像を学習
- 後半で細部の特徴を学習
- 効率的かつ高精度なモデルを構築

---

## 🧪 学習例

### 小規模データセット (各クラス100枚程度)

```bash
python train.py \
  --dataset dataset \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --save-every 10
```

### 中規模データセット (各クラス500-1000枚)

```bash
python train.py \
  --dataset dataset \
  --use-msa-net \
  --epochs 100 \
  --batch-size 16 \
  --lr 5e-5 \
  --save-every 10
```

### 大規模データセット (各クラス5000枚以上)

```bash
python train.py \
  --dataset dataset \
  --use-msa-net \
  --epochs 150 \
  --batch-size 32 \
  --lr 1e-5 \
  --resolution-threshold 0.01 \
  --save-every 5
```

---

## 🎯 推奨設定

### GPU別推奨バッチサイズ

| GPU | VRAM | 推奨バッチサイズ | 解像度 |
|-----|------|----------------|--------|
| RTX 3060 | 12GB | 8-16 | 384-512 |
| RTX 3080 | 10GB | 16-24 | 512-768 |
| RTX 3090 | 24GB | 24-32 | 768-1024 |
| RTX 4090 | 24GB | 32-48 | 1024 |

VRAM不足の場合:
```bash
# バッチサイズを減らす
python train.py --batch-size 4

# または勾配累積を使用 (カスタム実装が必要)
```

---

## 📈 モニタリング

### TensorBoardで確認できる項目

- **Loss/train**: 訓練ロス
- **Loss/val**: 検証ロス
- **Accuracy/train**: 訓練精度
- **Accuracy/val**: 検証精度
- **Confusion Matrix**: 混同行列
- **Learning_Rate**: 学習率推移
- **Resolution**: 現在の解像度

---

## 🐛 トラブルシューティング

### CUDA Out of Memory

```bash
# バッチサイズを減らす
python train.py --batch-size 4

# または解像度上昇を無効化 (固定384)
# train.pyの resolutions = [384] に変更
```

### モデルのダウンロードエラー

```bash
# 手動でキャッシュディレクトリを確認
ls ~/.cache/huggingface/hub

# ネットワークエラーの場合は再実行
```

### データセット読み込みエラー

```bash
# 画像ファイルの確認
find dataset -type f -name "*.jpg" | wc -l

# 破損ファイルの検出
python -c "
from PIL import Image
from pathlib import Path
for p in Path('dataset').rglob('*.jpg'):
    try:
        Image.open(p).verify()
    except:
        print(f'Corrupted: {p}')
"
```

---

## 📝 ライセンス

MIT License

---

## 🙏 謝辞

- **ConvNeXtV2**: Meta AI Research
- **timm**: Ross Wightman
- **PyTorch**: Meta AI

---

## 📮 お問い合わせ

Issues: GitHub Issues
