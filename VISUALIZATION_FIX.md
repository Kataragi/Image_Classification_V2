# 🔧 t-SNE可視化の改善 - 修正内容

## 問題の原因

inference.pyでスタイル空間可視化を実行した際、"Applying t-SNE dimensionality reduction..." というメッセージの後に処理が停止する問題がありました。

### 根本原因

1. **t-SNEの計算時間**: t-SNEは大量のデータ（特に1000サンプル以上）に対して非常に時間がかかる（O(n²)の計算複雑度）
2. **進捗表示の欠如**: `verbose`パラメータが設定されていなかったため、進捗が表示されず停止しているように見えた
3. **サンプリングなし**: データセットが大きい場合でも全データを処理しようとしていた
4. **代替手段なし**: より高速なPCAなどの代替方法が提供されていなかった

## 実装した修正

### 1. 進捗表示の追加

```python
# Before
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

# After
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=perplexity,
    n_iter=1000,
    verbose=2,  # 進捗表示を有効化
    n_jobs=-1   # すべてのCPUコアを使用
)
```

### 2. 自動サンプリング機能

5000サンプル以上のデータセットに対して自動的にサンプリング：

```python
max_samples = 5000  # コマンドラインで変更可能

if all_features.shape[0] > max_samples:
    print(f"  ⚠️  Dataset too large ({all_features.shape[0]} samples)")
    print(f"  Sampling {max_samples} samples for faster visualization...")
    # サンプリング処理...
```

### 3. PCAオプションの追加

高速な代替手段としてPCAのみの可視化を追加：

```python
if viz_method == 'pca':
    # PCA-only visualization (fast)
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(all_features)
else:
    # t-SNE (slower but better quality)
    # ...
```

### 4. 動的perplexity調整

データセットサイズに応じてperplexityを自動調整：

```python
n_samples = all_features.shape[0]
perplexity = min(30, max(5, n_samples // 100))
```

### 5. 詳細な進捗メッセージ

```python
print(f"  Total samples for visualization: {all_features.shape[0]}")
print(f"  Visualization method: {viz_method.upper()}")
print(f"    Step 1/2: PCA ({all_features.shape[1]}D → 50D)")
print(f"    Step 2/2: t-SNE (50D → 2D)")
print(f"    Samples: {n_samples}, Perplexity: {perplexity}")
print(f"    This may take a few minutes... ⏳")
```

## 新しいコマンドラインオプション

### --viz-method

可視化方法を選択：

```bash
# t-SNE (デフォルト、高品質だが遅い)
python inference.py --visualize --train-dataset dataset --viz-method tsne

# PCA (高速だが品質は低い)
python inference.py --visualize --train-dataset dataset --viz-method pca
```

### --max-samples

最大サンプル数を指定：

```bash
# 1000サンプルに制限（高速化）
python inference.py --visualize --train-dataset dataset --max-samples 1000

# 10000サンプルまで処理（時間かかる）
python inference.py --visualize --train-dataset dataset --max-samples 10000
```

## 使用例

### 推奨：高速PCA可視化

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --viz-method pca \
  --test-images test1.jpg test2.jpg \
  --output-viz visualizations/style_space_pca.png
```

**処理時間**: 数秒（データ量に関わらず）

### 高品質：t-SNE可視化（小規模データセット）

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --max-samples 1000 \
  --test-images test1.jpg test2.jpg \
  --output-viz visualizations/style_space_tsne.png
```

**処理時間**: 約3分（1000サンプル）

### 大規模データセット用

```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --visualize \
  --train-dataset dataset \
  --viz-method pca \
  --max-samples 2000 \
  --output-viz visualizations/style_space_fast.png
```

## パフォーマンス比較

| サンプル数 | PCA | t-SNE |
|-----------|-----|-------|
| 100 | < 1秒 | 約30秒 |
| 500 | < 1秒 | 約1分 |
| 1000 | < 1秒 | 約3分 |
| 5000 | 1-2秒 | 約10-15分 |
| 10000 | 2-3秒 | 約30-60分 |

## 進捗確認方法

### t-SNE実行中

以下のような出力が表示されます：

```
  Total samples for visualization: 1000
  Visualization method: TSNE
    Step 1/2: PCA (1024D → 50D)
    PCA explained variance: 87.34%
    Step 2/2: t-SNE (50D → 2D)
    Samples: 1000, Perplexity: 10
    This may take a few minutes... ⏳
[t-SNE] Computing 31 nearest neighbors...
[t-SNE] Indexed 1000 samples in 0.002s...
[t-SNE] Computed neighbors for 1000 samples in 0.051s...
[t-SNE] Computing pairwise distances...
[t-SNE] Computed conditional probabilities for sample 1000 / 1000
[t-SNE] Mean sigma: 0.893026
[t-SNE] KL divergence after 250 iterations with early exaggeration: 64.123456
[t-SNE] KL divergence after 1000 iterations: 1.234567
    ✅ t-SNE complete!
```

### PCA実行中

```
  Total samples for visualization: 5000
  Visualization method: PCA
    PCA (1024D → 2D)
    PCA explained variance: 68.45%
    ✅ PCA complete!
```

## トラブルシューティング

### "Iteration XX" で止まる

- **原因**: t-SNEの計算中（正常）
- **対処**: 数分待つか、Ctrl+Cで中断してPCAを使用

### メモリエラー

- **対処**: `--max-samples` を減らす

```bash
python inference.py --visualize --train-dataset dataset --max-samples 500
```

### 完全に停止（進捗なし）

- **原因**: サンプル数が多すぎる可能性
- **対処**: PCAを使用

```bash
python inference.py --visualize --train-dataset dataset --viz-method pca
```

## まとめ

この修正により：

✅ t-SNEの進捗がリアルタイムで確認可能
✅ 大規模データセットでも自動サンプリングで高速化
✅ PCAオプションで数秒で可視化完了
✅ より詳細なログメッセージ
✅ 柔軟なコマンドラインオプション

**推奨**: 初回はPCAで試し、結果を確認してからt-SNEを使用
