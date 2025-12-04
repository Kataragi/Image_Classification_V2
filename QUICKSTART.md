# ⚡ Quick Start Guide

このガイドに従って、5分で画風分類モデルのトレーニングを始められます。

## 📋 前提条件

- WSL2 (Ubuntu 22.04)
- NVIDIA GPU with CUDA 12.8
- Python 3.8以上

## 🚀 セットアップ (5ステップ)

### Step 1: リポジトリのクローン

```bash
cd ~/projects
git clone <your-repo-url> Image_Classification_V2
cd Image_Classification_V2
```

### Step 2: 仮想環境作成

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: 依存関係インストール

```bash
# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# その他のパッケージ
pip install -r requirements.txt
```

### Step 4: 環境確認

```bash
python verify_setup.py
```

すべてのチェックが✅なら次へ。

### Step 5: データセット準備

```bash
# ディレクトリ構造を作成
bash setup_dataset.sh

# 画像を配置
# dataset/anime/, dataset/brush/, 等に画像ファイルをコピー
```

## 🎓 トレーニング開始

### 基本的なトレーニング

```bash
python train.py \
  --dataset dataset \
  --epochs 50 \
  --batch-size 16
```

### MSA-Net使用 (高精度)

```bash
python train.py \
  --dataset dataset \
  --use-msa-net \
  --epochs 100 \
  --batch-size 16
```

### TensorBoard起動 (別ターミナル)

```bash
source venv/bin/activate
tensorboard --logdir logs
```

ブラウザで `http://localhost:6006` を開く。

## 🔮 推論

### 単一画像

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image test.jpg
```

### フォルダ一括

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --folder test_images/
```

### スタイル空間可視化

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --test-images img1.jpg img2.jpg
```

## 📊 トレーニング結果の確認

```bash
# チェックポイント確認
ls -lh checkpoints/

# TensorBoardログ確認
ls -lh logs/

# 可視化結果確認
ls -lh visualizations/
```

## 🎯 推奨設定

### 最小構成 (テスト用)

```bash
python train.py \
  --dataset dataset \
  --epochs 10 \
  --batch-size 8 \
  --save-every 5
```

### 標準構成

```bash
python train.py \
  --dataset dataset \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-every 10
```

### 高性能構成

```bash
python train.py \
  --dataset dataset \
  --use-msa-net \
  --epochs 150 \
  --batch-size 32 \
  --lr 5e-5 \
  --resolution-threshold 0.01 \
  --save-every 5
```

## 🐛 トラブルシューティング

### CUDA not available

```bash
# CUDA確認
nvidia-smi
nvcc --version

# PyTorchでCUDA確認
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

```bash
# バッチサイズを減らす
python train.py --batch-size 4
```

### モジュールが見つからない

```bash
# 仮想環境確認
which python

# 再インストール
pip install -r requirements.txt
```

## 📈 学習の進捗確認

### コマンドライン出力

```
Epoch 1/100 [Res: 384]: 100%|███████| 50/50 [00:30<00:00]
Epoch 1: Train Loss=1.2345, Train Acc=65.23%, Val Loss=1.1234, Val Acc=68.45%
```

### TensorBoard

- Loss推移
- 精度推移
- Confusion Matrix
- 学習率
- 解像度変化

## ✅ チェックリスト

トレーニング前:
- [ ] GPU認識確認 (`nvidia-smi`)
- [ ] Python環境確認 (`python verify_setup.py`)
- [ ] データセット配置完了
- [ ] 各クラス50枚以上の画像

トレーニング中:
- [ ] TensorBoard起動
- [ ] Loss減少を確認
- [ ] 定期的なチェックポイント保存

トレーニング後:
- [ ] best_model.pthが生成
- [ ] TensorBoardでConfusion Matrix確認
- [ ] テスト画像で推論確認

## 🎓 次のステップ

1. より多くのデータを集める
2. ハイパーパラメータ調整
3. MSA-Netの効果を検証
4. 異なる解像度で実験
5. 独自のデータセットで学習

## 💡 ヒント

- 初回は小さいデータセット(各50枚程度)で動作確認
- TensorBoardでリアルタイム確認を推奨
- Early Stoppingを使うと過学習を防げる
- バッチサイズはGPU VRAMに合わせて調整

---

**質問・問題があれば**: GitHub Issuesへ
