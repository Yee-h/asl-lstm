import h5py
import os

file_path = r'd:\Document\Code\process_wsasl\pose_action_dataset\WLASL100\WLASL100_135-Train.hdf5'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with h5py.File(file_path, 'r') as f:
    print("Keys:", list(f.keys())[:5])
    key = list(f.keys())[0]
    grp = f[key]
    print(f"Group: {key}")
    for k in grp.keys():
        ds = grp[k]
        if isinstance(ds, h5py.Dataset):
            print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}")
            if k == 'data':
                print(f"    First frame first point: {ds[0, :, 0]}")
        else:
            print(f"  {k} (Group)")
