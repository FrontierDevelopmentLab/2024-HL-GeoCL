"""
DEPRECATED: This module is an earlier iteration of the SHEATH model.

The active SHEATH implementation is in geocloak.models.sheath, which is
compatible with the trained checkpoint (sheath_best_model_checkpoint_1.pth).

Key differences from the current version:
- This version uses BatchNorm layers; the current version does not.
- This version uses StandardScaler for targets; the current uses MinMaxScaler.
- The checkpoint was trained WITHOUT BatchNorm, so this model variant
  cannot load it.

This module is retained for historical reference only.
"""
