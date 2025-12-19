import os
import pickle
import numpy as np

MAX_FRAMES = 300

def generate_split(benchmark, split_name):
    pkl_path = './data/NTU120-RGB-D/ntu120_3d.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    out_dir = f'./data/NTU120-RGB-D/{benchmark}'
    os.makedirs(out_dir, exist_ok=True)
    
    for mode in ['train', 'val']:
        split_key = f'{split_name}_{mode}'
        ids = data['split'][split_key]
        data_list = []
        sample_names = []
        label_list = []
        
        for ann in data['annotations']:
            if ann['frame_dir'] in ids:
                kp = ann['keypoint']  # (M, T, V, C)
                M, T, V, C = kp.shape
                # Pad T to MAX_FRAMES
                if T < MAX_FRAMES:
                    pad = np.zeros((M, MAX_FRAMES - T, V, C))
                    kp = np.concatenate([kp, pad], axis=1)
                elif T > MAX_FRAMES:
                    kp = kp[:, :MAX_FRAMES, :, :]
                # Pad M to 2
                if M < 2:
                    pad = np.zeros((2 - M, MAX_FRAMES, V, C))
                    kp = np.concatenate([kp, pad], axis=0)
                kp = kp.transpose(3, 1, 2, 0)  # (C, T, V, M)
                data_list.append(kp)
                sample_names.append(ann['frame_dir'])
                label_list.append(ann['label'])
        
        data_arr = np.stack(data_list)  # Now all same shape
        np.save(f'{out_dir}/{mode}_data.npy', data_arr)
        with open(f'{out_dir}/{mode}_label.pkl', 'wb') as f:
            pickle.dump((sample_names, label_list), f)
        print(f"[{benchmark.upper()}] {mode}: {len(data_list)} samples → {out_dir}/{mode}_data.npy ({data_arr.shape})")

if __name__ == '__main__':
    generate_split('xsub', 'xsub')
    generate_split('xset', 'xset')
