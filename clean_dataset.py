#!/usr/bin/env python3
"""Clean dataset by removing defect classes and adjusting ripeness class IDs."""
import os
from pathlib import Path

# Dataset root
DATASET_ROOT = "data/temp"

# Original: ['Defects-black spots'=0, 'Defects-white fungus'=1, 'overripe'=2, 'ripe'=3, 'unripe'=4]
# New: ['overripe'=0, 'ripe'=1, 'unripe'=2]
# Mapping: old class -> new class (or None to remove)
CLASS_MAPPING = {
    0: None,  # remove Defects-black spots
    1: None,  # remove Defects-white fungus
    2: 0,     # overripe: 2 -> 0
    3: 1,     # ripe: 3 -> 1
    4: 2,     # unripe: 4 -> 2
}


def clean_label_file(label_path):
    """Read label file, filter and remap classes, write back."""
    if not os.path.exists(label_path):
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue  # skip malformed
        
        old_class = int(parts[0])
        new_class = CLASS_MAPPING.get(old_class)
        
        if new_class is None:
            # skip this annotation (defect class)
            continue
        
        # replace class ID
        parts[0] = str(new_class)
        new_lines.append(' '.join(parts))
    
    # write back
    with open(label_path, 'w') as f:
        f.write('\n'.join(new_lines))
        if new_lines:
            f.write('\n')


def main():
    root = Path(DATASET_ROOT)
    splits = ['train', 'valid', 'test']
    
    total_files = 0
    for split in splits:
        labels_dir = root / split / 'labels'
        if not labels_dir.exists():
            print(f"Skipping {split} (no labels dir)")
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        print(f"Cleaning {len(label_files)} files in {split}/labels...")
        
        for lf in label_files:
            clean_label_file(lf)
            total_files += 1
    
    print(f"Done. Cleaned {total_files} label files.")
    print("Classes removed: 'Defects-black spots' (0), 'Defects-white fungus' (1)")
    print("New mapping: overripe=0, ripe=1, unripe=2")


if __name__ == '__main__':
    main()
