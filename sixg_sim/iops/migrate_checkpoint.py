"""
Checkpoint migration: 51-dim → 59-dim observation space.

Zero-pads the first linear layer weights to accommodate the
8 new Multi-eNB IOPS observation dimensions added to Block C.
"""

import torch
from pathlib import Path


OLD_OBS_DIM = 51
NEW_OBS_DIM = 59
DELTA = NEW_OBS_DIM - OLD_OBS_DIM   # 8


def migrate_checkpoint(src_path: str, dst_path: str) -> bool:
    """
    Convert a 51-dim policy checkpoint to 59-dim.

    Args:
        src_path: path to old .pt checkpoint
        dst_path: path to save migrated checkpoint
    Returns:
        True if migration succeeded
    """
    try:
        ckpt = torch.load(src_path, map_location='cpu', weights_only=False)

        # Handle both raw state_dict and wrapped checkpoint
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt

        migrated = {}
        for key, tensor in sd.items():
            if 'fc1.weight' in key and tensor.shape[1] == OLD_OBS_DIM:
                # Zero-pad columns: (H, 51) → (H, 59)
                pad = torch.zeros(tensor.shape[0], DELTA)
                migrated[key] = torch.cat([tensor, pad], dim=1)
                print(f"  Migrated {key}: {tensor.shape} -> "
                      f"{migrated[key].shape}")
            else:
                migrated[key] = tensor

        if 'state_dict' in ckpt:
            ckpt['state_dict'] = migrated
            ckpt.setdefault('metadata', {})['migrated_from'] = OLD_OBS_DIM
            ckpt['metadata']['migrated_to'] = NEW_OBS_DIM
            torch.save(ckpt, dst_path)
        else:
            torch.save(migrated, dst_path)

        print(f"Migration complete: {src_path} -> {dst_path}")
        return True
    except Exception as e:
        print(f"Migration failed: {e}")
        return False


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: python -m sixg_sim.iops.migrate_checkpoint "
              f"<src.pt> <dst.pt>")
        sys.exit(1)
    ok = migrate_checkpoint(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
