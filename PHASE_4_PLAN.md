# Phase 4: Supervised Learning - Ready to Start

## Current State

✅ **272/272 tests passing**  
✅ **All Phase 3 components integrated**  
✅ **Synthetic battle generation pipeline ready**  
✅ **Feature encoding and model architecture complete**  

## What's Ready for Training

### Data Pipeline
- **BattleGenerator**: Generates synthetic battles from heuristic agents
- **BattleReplayer**: Iterates through battle trajectories
- **FeatureEncoder**: Converts observations → PyTorch tensors
- **Champions MB team pool**: 50 realistic teams for sampling

### Model Architecture
- **TransformerPolicy**: 4-layer Transformer with 5 prediction heads
- **Multi-task loss**: Main (action) + 4 auxiliary tasks
- **Action masking**: Ensures legal moves only
- **Inference utilities**: Sampling and deterministic action selection

### Regulation & Rules
- **Champions MB config**: 635+ legal Pokemon, VGC 2020 rules
- **Level 50 doubles format**
- **13 restricted Pokemon**

## Phase 4 Tasks

### Task 1: Data Generation & Loading (2-3 hours)
Create `src/training/data.py`:
- Generate N synthetic battles (e.g., 10k-100k transitions)
- Create PyTorch Dataset/DataLoader:
  * Input: Encoded battle state (from FeatureEncoder)
  * Output: (action_pair, win_prob, opponent_moves, opponent_items, battle_length)
  * Batch collation with padding/masking
- Save/load generated data to disk (zarr or HDF5)
- Track dataset statistics (action distributions, win rates, etc.)

### Task 2: Training Loop (2-3 hours)
Create `src/training/train.py`:
- Initialize TransformerPolicy and optimizer (Adam)
- Training loop:
  * Forward pass through model
  * Compute multi-task loss (action + 4 auxiliary)
  * Backward pass
  * Gradient clipping (optional)
  * Optimizer step
- Validation loop:
  * Action prediction accuracy
  * Win probability calibration
  * Auxiliary task metrics
- Checkpoint saving (best model + latest)
- Learning rate scheduling
- Early stopping on validation loss

### Task 3: Logging & Experiments (1-2 hours)
Create `scripts/train.py`:
- CLI arguments:
  * `--epochs`: Training epochs
  * `--batch_size`: Batch size (32-64)
  * `--num_battles`: Synthetic battles to generate
  * `--seed`: Random seed for reproducibility
  * `--wandb`: Enable WandB logging
- Generate synthetic data
- Run training with logging
- Save final checkpoint
- Log final metrics

### Task 4: Evaluation (1 hour)
Create `scripts/eval.py`:
- Load trained model
- Run on held-out validation set
- Report metrics:
  * Action prediction accuracy
  * Win probability calibration (Brier score)
  * Opponent moveset accuracy
  * Battle length prediction error
- Optional: Play against heuristic agents, report win rate

## Expected Performance (Baseline)

After Phase 4 (supervised learning):
- **Action accuracy**: 60-75% (lower because of 126 action space + randomness)
- **Win probability**: ~70% calibration (auxiliary task helps)
- **Opponent movesets**: 40-50% (hard to predict without game history)
- **Win rate vs heuristics**: 50-70% (still playing safe patterns from data)

This is the **imitation baseline** — good enough to transition to RL in Phase 5.

## Recommended Implementation Choices

1. **Loss weights** for multi-task learning:
   ```python
   main_weight = 1.0          # Action prediction
   win_prob_weight = 0.1      # Win probability
   moves_weight = 0.05        # Opponent movesets
   items_weight = 0.05        # Opponent items
   length_weight = 0.05       # Battle length
   ```

2. **Batch size**: 32 (good for fast iteration)

3. **Learning rate**: 1e-3 with linear warmup (10% of training)

4. **Epochs**: Start with 5-10, monitor validation loss

5. **Data**: Generate 10k-50k battles initially (can scale up)

## Files to Create/Modify

```
src/training/
  ├─ data.py           (NEW: DataLoader, dataset generation)
  ├─ train.py          (NEW: Training loop implementation)
  └─ metrics.py        (NEW: Evaluation metrics)

scripts/
  ├─ train.py          (NEW: CLI training script)
  ├─ eval.py           (NEW: Evaluation script)
  └─ generate_data.py  (NEW: Generate synthetic battles)

configs/experiments/
  └─ supervised_v1.yaml (NEW: Experiment config)
```

## Success Criteria for Phase 4

- ✅ Generate 10k+ deterministic synthetic battles
- ✅ Train for 5+ epochs without crashes
- ✅ Validation loss decreases (sign of learning)
- ✅ Action accuracy > 50% on held-out data
- ✅ All metrics logged to WandB (or CSV)
- ✅ Checkpoint saved and can reload
- ✅ Win rate vs random agent > 50%

## Transition to Phase 5

Once Phase 4 completes with a trained model:
1. Use as initialization for PPO agent (Phase 5)
2. Fine-tune with RL in self-play
3. Add opponent pool curriculum
4. Measure improvement over imitation baseline

---

**Ready to proceed with Phase 4 implementation?**

Implementation should prioritize:
1. Getting a working training loop first (even if slow)
2. Measuring that learning happens (val loss down)
3. Then optimize (speed, metrics, logging)
