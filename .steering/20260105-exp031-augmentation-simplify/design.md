# Design

## アプローチ

### Augmentation設定変更

**Before (exp028/029/030):**
```yaml
augmentation:
  train:
    image_size: 512
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation_limit: 20
    shift_limit: 0.1
    scale_limit: 0.2
    elastic_transform: true
    brightness_contrast: true
    gamma_transform: true
    gaussian_noise: true
    blur: true
    mixup:
      enabled: true
      alpha: 0.2
      prob: 0.5
```

**After (exp031):**
```yaml
augmentation:
  train:
    image_size: 512
    horizontal_flip: 0.5
    vertical_flip: 0.5
    brightness_contrast: true
    # 以下すべて無効化
    rotation_limit: 0
    shift_limit: 0.0
    scale_limit: 0.0
    elastic_transform: false
    gamma_transform: false
    gaussian_noise: false
    blur: false
    mixup:
      enabled: false
```

## 変更コンポーネント

| ファイル | 変更内容 |
|---------|---------|
| config/exp031.yaml | augmentationセクション変更 |

## 影響範囲
- configのみ変更
- コードの変更なし
