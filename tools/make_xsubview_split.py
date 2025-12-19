# tools/make_xsubview_split.py   ← 100% working version with prints
import os
import pickle
import numpy as np

train_subjects = [1,2,4,5,8,9,13,14,15,16]          # 10 subjects
test_subjects  = list(range(17,47))                 # 30 unseen subjects

with open('./data/NTU120-RGB-D/ntu120_3d.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

out_dir = './data/NTU120-RGB-D/xsubview'
os.makedirs(out_dir, exist_ok=True)

for mode, subs, cams in [('train', train_subjects, [2,3]), ('val', test_subjects, [1])]:
    samples = []
    names   = []
    labels  = []
    
    for ann in data['annotations']:
        sub = int(ann['frame_dir'][9:12])   # P001 → 1
        cam = int(ann['frame_dir'][5:8])    # C001 → 1
        if sub in subs and cam in cams:
            kp = ann['keypoint']                     # (M,T,V,C)
            M, T, V, C = kp.shape
            # pad / truncate to 300 frames
            if T < 300:
                kp = np.pad(kp, ((0,0),(0,300-T),(0,0),(0,0)))
            elif T > 300:
                kp = kp[:, :300, :, :]
            # pad to 2 performers
            if M < 2:
                kp = np.pad(kp, ((0,2-M),(0,0),(0,0),(0,0)))
            kp = kp.transpose(3,1,2,0)                  # (C,T,V,M)
            samples.append(kp)
            names.append(ann['frame_dir'])
            labels.append(ann['label'])
    
    data_arr = np.stack(samples)
    np.save(f'{out_dir}/{mode}_data.npy', data_arr)
    with open(f'{out_dir}/{mode}_label.pkl', 'wb') as f:
        pickle.dump((names, labels), f)
    
    print(f"Created → {out_dir}/{mode}_data.npy   ({data_arr.shape})")
    print(f"Created → {out_dir}/{mode}_label.pkl ({len(names)} samples)")

print("All files created successfully!")