# import pickle, random, os

# # Input files
# train_pkl = "./data/NTU-RGB-D/xsub/train_label.pkl"
# val_pkl = "./data/NTU-RGB-D/xsub/val_label.pkl"
# output_train_pkl = "./data/NTU-RGB-D/xsub/random10_train_label.pkl"
# output_val_pkl = "./data/NTU-RGB-D/xsub/random10_val_label.pkl"

# random.seed(42)

# # Load both label sets
# with open(train_pkl, 'rb') as f:
#     train_names, train_labels = pickle.load(f)
# with open(val_pkl, 'rb') as f:
#     val_names, val_labels = pickle.load(f)

# # Merge them for consistent subject IDs
# all_names = train_names + val_names
# all_labels = train_labels + val_labels

# # Identify all unique subjects
# subjects = sorted({int(n.split('S')[1].split('C')[0]) for n in all_names})

# # Randomly select 7 (or 10) for training
# train_subjects = set(random.sample(subjects, 7))
# val_subjects = set(subjects) - train_subjects

# # Split
# train_s, train_l, val_s, val_l = [], [], [], []
# for name, label in zip(all_names, all_labels):
#     sid = int(name.split('S')[1].split('C')[0])
#     if sid in train_subjects:
#         train_s.append(name)
#         train_l.append(label)
#     else:
#         val_s.append(name)
#         val_l.append(label)

# # Save new files
# os.makedirs(os.path.dirname(output_train_pkl), exist_ok=True)
# with open(output_train_pkl, 'wb') as f:
#     pickle.dump((train_s, train_l), f)
# with open(output_val_pkl, 'wb') as f:
#     pickle.dump((val_s, val_l), f)

# print("✅ New split created successfully.")
# print(f"Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
# print(f"Val subjects   ({len(val_subjects)}): {sorted(val_subjects)}")
# print(f"Train samples: {len(train_s)}")
# print(f"Val samples:   {len(val_l)}")

import os
import pickle
import random
import numpy as np
from numpy.lib.format import open_memmap

# Configuration
NUM_TRAIN_SUBJECTS = 10          # set to 7 or 10 as you like
SEED = 42
ROOT = "./data/NTU-RGB-D/xsub"

# Input files (original NTU xsub split)
TRAIN_DATA = f"{ROOT}/train_data.npy"
TRAIN_LABEL = f"{ROOT}/train_label.pkl"
VAL_DATA   = f"{ROOT}/val_data.npy"
VAL_LABEL  = f"{ROOT}/val_label.pkl"

# Output files (random split)
OUT_TRAIN_LABEL = f"{ROOT}/random_train_label.pkl"
OUT_VAL_LABEL   = f"{ROOT}/random_val_label.pkl"
OUT_TRAIN_DATA  = f"{ROOT}/random_train_data.npy"
OUT_VAL_DATA    = f"{ROOT}/random_val_data.npy"

random.seed(SEED)

# 1) Load original labels (names + labels) from both splits
with open(TRAIN_LABEL, "rb") as f:
    train_names, train_labels = pickle.load(f)
with open(VAL_LABEL, "rb") as f:
    val_names, val_labels = pickle.load(f)

n_train_src = len(train_names)
n_val_src = len(val_names)

# Combined view over names/labels to define subject split consistently
all_names  = train_names + val_names
all_labels = train_labels + val_labels

# Helper: extract subject id from sample name like 'S001C001P001R001A001'
def subj_id(name: str) -> int:
    return int(name.split('S')[1].split('C')[0])

subjects = sorted({subj_id(n) for n in all_names})
if NUM_TRAIN_SUBJECTS > len(subjects):
    raise ValueError(f"Requested {NUM_TRAIN_SUBJECTS} subjects but only {len(subjects)} exist.")

train_subjects = set(random.sample(subjects, NUM_TRAIN_SUBJECTS))
val_subjects = set(subjects) - train_subjects

# 2) Build new label lists AND keep track of indices into the combined data
new_train_names, new_train_labels, new_train_indices = [], [], []
new_val_names,   new_val_labels,   new_val_indices   = [], [], []

for idx, (n, l) in enumerate(zip(all_names, all_labels)):
    sid = subj_id(n)
    if sid in train_subjects:
        new_train_names.append(n)
        new_train_labels.append(l)
        new_train_indices.append(idx)
    else:
        new_val_names.append(n)
        new_val_labels.append(l)
        new_val_indices.append(idx)

# Save new label pkls
with open(OUT_TRAIN_LABEL, "wb") as f:
    pickle.dump((new_train_names, new_train_labels), f)
with open(OUT_VAL_LABEL, "wb") as f:
    pickle.dump((new_val_names, new_val_labels), f)

print("Labels written:")
print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
print(f"  Val subjects   ({len(val_subjects)}): {sorted(val_subjects)}")
print(f"  Train samples: {len(new_train_names)}")
print(f"  Val samples:   {len(new_val_names)}")

# 3) Create matching data arrays by copying rows from original train/val .npy
#    We reference the combined index space:
#       0..n_train_src-1  -> rows from TRAIN_DATA
#       n_train_src..     -> rows from VAL_DATA (offset by n_train_src)

# Open source arrays as memory-mapped to avoid loading everything at once
src_train = np.load(TRAIN_DATA, mmap_mode="r")
src_val   = np.load(VAL_DATA,   mmap_mode="r")

sample_shape = src_train.shape[1:]   # [C, T, V, M]
dtype = src_train.dtype

# Sanity check: src_val must have same per-sample shape and dtype
assert src_val.shape[1:] == sample_shape
assert src_val.dtype == dtype

# Prepare destination memmaps
dst_train = open_memmap(OUT_TRAIN_DATA, mode="w+", dtype=dtype,
                        shape=(len(new_train_indices),) + sample_shape)
dst_val   = open_memmap(OUT_VAL_DATA,   mode="w+", dtype=dtype,
                        shape=(len(new_val_indices),) + sample_shape)

def copy_rows(indices, dst):
    write_pos = 0
    for idx in indices:
        if idx < n_train_src:
            # from TRAIN_DATA
            dst[write_pos] = src_train[idx]
        else:
            # from VAL_DATA
            src_idx = idx - n_train_src
            dst[write_pos] = src_val[src_idx]
        write_pos += 1

print("Writing random_train_data.npy ...")
copy_rows(new_train_indices, dst_train)
print("Writing random_val_data.npy ...")
copy_rows(new_val_indices,   dst_val)

# Flush to disk
del dst_train
del dst_val
print("Data arrays written successfully.")
