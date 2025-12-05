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

### チェックポイントからの再開

学習を中断した場合や、継続学習したい場合：

```bash
python train.py \
  --dataset dataset \
  --resume checkpoints/checkpoint_epoch_50.pth \
  --epochs 150
```

**自動的に復元される情報:**
- モデルの重み
- オプティマイザの状態
- 現在のエポック
- 現在の解像度
- Best validation loss
- Validation loss履歴

**利点:**
- 学習が中断されても安全に再開できる
- 解像度の段階的学習も継続される
- TensorBoardのログも連続して記録される

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
| `--resolution-threshold` | Val Loss上昇時の解像度変更閾値 | `0.1` |
| `--save-every` | 保存間隔(エポック) | `10` |
| `--early-stop-on-increase` | Val Loss上昇で停止 | `False` |
| `--output-dir` | 出力ディレクトリ | `checkpoints` |
| `--resume` | 再開するチェックポイント | `None` |

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

**基本的な可視化 (t-SNE):**
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --test-images test1.jpg test2.jpg test3.jpg \
  --output-viz visualizations/style_space.png
```

**高速な可視化 (PCA):**
t-SNEが遅い場合はPCAを使用：
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --viz-method pca \
  --test-images test1.jpg test2.jpg \
  --output-viz visualizations/style_space_pca.png
```

**大規模データセット用:**
サンプル数を調整して高速化：
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --max-samples 2000 \
  --viz-method pca \
  --output-viz visualizations/style_space_fast.png
```

**カスタム軸範囲:**
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

**可視化オプション:**
- `--viz-method tsne`: t-SNE使用 (デフォルト、高品質だが遅い)
- `--viz-method pca`: PCA使用 (高速だが品質は低い)
- `--max-samples N`: 最大サンプル数 (デフォルト: 5000)

**注意**: スタイル空間可視化では、見やすさのため原色系の鮮やかな色を使用しています

---

## 📊 段階的解像度学習について

### 新しい解像度上昇メカニズム（Val Loss スパイク検出）

トレーニング中にVal Lossが大きく上昇した場合、自動的に解像度を上げて学習を継続します：

**動作:**
1. **Val Loss監視**: 各エポック後にVal Lossを前エポックと比較
2. **スパイク検出**: `--resolution-threshold`（デフォルト0.1）以上の上昇を検出
3. **自動ロールバック**: 悪化したエポックの学習を破棄し、前エポックのモデルに戻る
4. **解像度上昇**: 次の解像度段階（384→512→768→1024）に移行
5. **学習継続**: ロールバックしたモデルから高解像度で学習を再開

**例:**
```
Epoch 45: Val Loss = 0.250  ✓ Good
Epoch 46: Val Loss = 0.380  ❌ Spike detected! (increase: 0.130 > 0.1)
→ Epoch 46を破棄、Epoch 45のモデルに戻る
→ 解像度を384→512に上昇
→ Epoch 46から512x512で再トレーニング
```

**利点:**
- 過学習や学習の不安定性を早期に検出
- 悪化したエポックを無駄にしない
- 高解像度で新しい特徴を学習して回復
- より安定したトレーニング

**カスタマイズ:**
```bash
# 閾値を厳しく（小さな上昇でも解像度変更）
python train.py --resolution-threshold 0.05

# 閾値を緩く（大きな上昇のみ反応）
python train.py --resolution-threshold 0.2
```

**注意:**
- 最大解像度（1024x1024）に達している場合は解像度を上げずに継続
- `--early-stop-on-increase` とは独立して動作（early-stopは学習終了、thresholdは解像度変更）

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

### t-SNE可視化が遅い/停止する

t-SNEは大量のデータに対して非常に時間がかかります（数分〜数十分）。

**解決策:**

1. **PCAを使用** (推奨・最速):
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --viz-method pca
```

2. **サンプル数を減らす**:
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --max-samples 1000
```

3. **進捗を確認**:
   - t-SNEは `verbose=2` で進捗が表示されます
   - "Iteration XX" というメッセージが出ていれば動作中です
   - 完全に停止している場合は Ctrl+C で中断してPCAを試してください

**参考処理時間:**
- 500サンプル (t-SNE): 約1分
- 1000サンプル (t-SNE): 約3分
- 5000サンプル (t-SNE): 約10-15分
- 任意のサンプル数 (PCA): 数秒

---

## 📜 更新履歴

### Version 2.2 (Latest)

**重要な変更:**

1. **解像度上昇メカニズムの刷新**（`--resolution-threshold`）
   - **旧仕様**: 3エポック連続でval_lossの改善が閾値以下の場合に解像度上昇
   - **新仕様**: val_lossが閾値以上上昇した場合に自動ロールバック+解像度上昇
   - デフォルト値変更: `0.02` → `0.1`

   **動作:**
   - Val lossスパイク（過学習など）を検出
   - 悪化したエポックの学習を破棄
   - 前エポックのモデルに自動ロールバック
   - 解像度を上げて学習継続

   **利点:**
   - 過学習や不安定な学習を早期に検出・回復
   - 悪化したエポックを無駄にしない
   - より堅牢なトレーニング

   **使用例:**
   ```bash
   # デフォルト（0.1以上の上昇で解像度変更）
   python train.py --dataset dataset

   # より敏感に（0.05以上で反応）
   python train.py --dataset dataset --resolution-threshold 0.05
   ```

**破壊的変更:**
- `--resolution-threshold` の意味が完全に変更されました
- 旧バージョンの挙動を期待する場合は注意が必要です

### Version 2.1

**新機能:**

1. **チェックポイントからの学習再開機能** (`--resume`)
   - 学習が中断されても途中から再開可能
   - モデル、オプティマイザ、エポック、解像度などの状態を完全復元
   - 段階的解像度学習も正しく継続
   - 使用例: `python train.py --resume checkpoints/checkpoint_epoch_50.pth`

2. **t-SNE可視化の大幅なパフォーマンス改善**
   - 進捗表示の追加（`verbose=2`）で処理状況が可視化
   - 5000サンプル以上のデータセットは自動サンプリング
   - PCAオプション追加で数秒で可視化完了（`--viz-method pca`）
   - サンプル数制限オプション（`--max-samples`）
   - マルチコアCPU対応（`n_jobs=-1`）

3. **可視化の色改善**
   - パステルカラーから原色系の鮮やかな色に変更
   - クラス間の区別がより明確に

**パフォーマンス:**
- PCA可視化: 任意のサンプル数で数秒
- t-SNE可視化: 1000サンプルで約3分（以前より高速化）
- 大規模データセットでも快適に動作

**ドキュメント:**
- VISUALIZATION_FIX.md: 可視化改善の詳細説明を追加
- README.md: チェックポイント再開、可視化オプションの説明を追加
- QUICKSTART.md: 高速PCA可視化例を追加

### Version 2.0

**初期リリース:**
- ConvNeXtV2-Base-22k-384ベースの画風分類
- オプションのMSA-Net統合
- 段階的解像度学習（384→512→768→1024）
- 自動クラス重み付け
- TensorBoard統合
- 8クラス分類対応

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
