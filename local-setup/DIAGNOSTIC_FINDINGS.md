# Diagnostic Findings: Why Val Dice is 56.9%

Date: 2025-10-07
Best Val Dice: **0.5690** (56.9%) - Far below 70-80% healthy baseline

## Summary

**The model IS learning** (Train Dice improved from 24% → 79%), but validation performance is poor (Val Dice only 57%). This indicates a **fundamental training issue**, not a data problem.

## Findings

### ✅ Dataset is CORRECT
- Total samples: 1,316
- Positive (has pneumothorax): 291 (22.1%)
- Negative (no pneumothorax): 1,025 (77.9%)
- **This ratio is normal for medical imaging**
- Empty masks are legitimate negative samples

### ✅ Dice Metric has Minor Bug
- Test 1 (Perfect match): ✅ PASS (1.0000)
- Test 2 (No match): ✅ PASS (0.0000)
- Test 3 (50% overlap): ❌ FAIL (got 0.5000, expected 0.6667)
- Test 4 (Small perfect match): ✅ PASS (1.0000)

**Issue**: Dice calculation in test might be using different formula than training.
**Impact**: Metrics reporting may be slightly off, but not the root cause.

### ⚠️ Training Pattern Analysis

Epoch-by-epoch progression shows **severe overfitting**:

| Epoch | Train Dice | Val Dice | Gap | Status |
|-------|-----------|----------|-----|---------|
| 1 | 24.6% | 18.9% | 5.7% | Learning |
| 5 | 59.0% | 39.8% | 19.2% | Overfitting starts |
| 10 | 69.6% | 41.1% | 28.5% | Severe overfitting |
| 12 | 72.7% | **52.4%** | 20.3% | Best Val Dice |
| 18 | 78.8% | **56.9%** | 21.9% | Best overall (checkpoint) |
| 20 | 78.7% | 51.7% | 27.0% | Degrading |

**Key observations**:
1. ✅ Model CAN learn (Train Dice reached 79%)
2. ❌ Huge train/val gap (22-28%) = severe overfitting
3. ❌ Val Dice plateaus around 40-57%, never breaks through
4. ⚠️ Val Dice actually PEAKED at epoch 12 (52.4%), then fluctuated

### 🔍 Root Cause Hypotheses

Based on the training pattern, likely causes (in priority order):

#### 1. **Class Imbalance Not Properly Handled** (Most Likely)
**Evidence**:
- Dataset has 77.9% negative samples
- WeightedRandomSampler with POS_SAMPLE_WEIGHT=10.0
- But batches may still be heavily negative-biased

**Why this causes 56.9% Val Dice**:
- Model learns to predict negative (no pneumothorax) very well
- Model struggles with positive cases (only 22% of data)
- Val Dice gets stuck because it can't generalize to positives

**Test**: Check actual batch composition during training
- Expected: ~50% positive batches (due to weighting)
- Reality: Probably still 70%+ negative batches

#### 2. **Loss Function Imbalance**
**Evidence**:
- Using BCE:1 + Dice:1 (equal weights)
- Competition used BCE:3 + Dice:1 + Focal:4

**Why this causes low Val Dice**:
- BCE loss doesn't focus on hard examples
- Dice loss alone isn't enough for severe class imbalance
- Focal loss specifically targets hard-to-classify pixels

**Test**: Switch to competition loss weights

#### 3. **Model Capacity / Architecture Issue**
**Evidence**:
- Using ResNet-34 encoder (24M parameters)
- Competition also used ResNet-34

**Why unlikely**:
- Same architecture as competition
- Train Dice 79% shows model CAN learn
- Problem is generalization, not capacity

#### 4. **Data Augmentation Still Insufficient**
**Evidence**:
- Added elastic transforms (+5% improvement 52% → 57%)
- But still 25% below target

**Why this causes low Val Dice**:
- Only 291 positive samples to learn from
- Even with augmentation, not enough variation
- Model memorizes training pneumothorax patterns

#### 5. **Threshold Issue**
**Evidence**:
- Using threshold=0.5 for binary prediction
- Competition may have used different threshold

**Test**: Try thresholds 0.3, 0.4, 0.6, 0.7

## Recommended Diagnostic Steps

### Step 1: Check Batch Composition (5 min)
```python
# Add this to your notebook during training
for epoch in range(1):
    for batch_idx, (images, masks) in enumerate(train_loader):
        positive_pixels = (masks > 0.5).float().sum().item()
        total_pixels = masks.numel()
        pct = positive_pixels / total_pixels * 100
        print(f"Batch {batch_idx}: {pct:.2f}% positive pixels")
        if batch_idx >= 10:
            break
```

**Expected**: 5-15% positive pixels per batch (due to small pneumothorax area)
**If <1%**: WeightedRandomSampler not working
**If >20%**: Over-sampling positives (also bad)

### Step 2: Test Different Thresholds (10 min)
```python
# After training, evaluate at different thresholds
model.eval()
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for thresh in thresholds:
    val_dice_scores = []
    for images, masks in val_loader:
        outputs = model(images.cuda())
        preds = (torch.sigmoid(outputs) > thresh).float()
        dice = dice_coefficient(preds, masks.cuda(), threshold=thresh)
        val_dice_scores.append(dice)
    print(f"Threshold {thresh}: Mean Val Dice = {np.mean(val_dice_scores):.4f}")
```

### Step 3: Switch to Focal Loss (30 min)
```python
# Try competition loss function
from segmentation_models_pytorch.losses import FocalLoss

focal_loss = FocalLoss(mode='binary')

def combo_loss_v2(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    focal = focal_loss(pred, target)
    return 3 * bce + 1 * dice + 4 * focal  # Competition weights
```

### Step 4: Visualize Predictions (15 min)
```python
# Check what model actually predicts
model.eval()
images, masks = next(iter(val_loader))
outputs = model(images.cuda())
preds = torch.sigmoid(outputs)

for i in range(4):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title('Input')

    plt.subplot(1, 4, 2)
    plt.imshow(masks[i].squeeze(), cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 4, 3)
    plt.imshow(preds[i].cpu().squeeze(), cmap='hot', vmin=0, vmax=1)
    plt.title('Prediction (prob)')

    plt.subplot(1, 4, 4)
    plt.imshow((preds[i] > 0.5).cpu().squeeze(), cmap='gray')
    plt.title('Prediction (binary)')
    plt.show()
```

## Expected Outcomes

### If Issue is Class Imbalance:
- Batches will show <1% positive pixels
- **Fix**: Increase POS_SAMPLE_WEIGHT to 20.0 or 50.0
- **Expected improvement**: +10-15% Val Dice

### If Issue is Loss Function:
- Switching to Focal loss should help with hard examples
- **Expected improvement**: +5-10% Val Dice

### If Issue is Threshold:
- Optimal threshold might be 0.3 or 0.4 instead of 0.5
- **Expected improvement**: +3-8% Val Dice (just from better threshold)

### If Issue is Insufficient Data:
- No quick fix - need competition's 4-stage pipeline
- **Expected improvement**: Requires full pipeline

## Bottom Line

**56.9% Val Dice is NOT acceptable** for any reasonable implementation. The fact that:
- ✅ Train Dice = 79% (model can learn)
- ❌ Val Dice = 57% (model can't generalize)
- ❌ Gap = 22% (severe overfitting)

...indicates **class imbalance** is not being handled properly, OR the **loss function** is wrong.

**Next steps**:
1. Run Step 1 (batch composition check) - **DO THIS FIRST**
2. If batches are negative-heavy, increase POS_SAMPLE_WEIGHT to 20-50
3. If batches look OK, switch to Focal loss (Step 3)
4. Re-train with fixes and target 70-75% Val Dice minimum

**DO NOT proceed with full competition pipeline until Val Dice ≥ 70%.**
