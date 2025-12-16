import os
import shutil
import glob

# Define directory paths
train_dir = 'data/1D_custom/processed/train'
val_dir = 'data/1D_custom/processed/val'

os.makedirs(val_dir, exist_ok=True)
shutil.rmtree(val_dir)
os.makedirs(val_dir)

pkl_files = sorted(glob.glob(os.path.join(train_dir, '*.pkl')))

# If no .pkl files found, exit directly
if not pkl_files:
    print("No .pkl files found in the train directory.")
else:
    total = len(pkl_files)
    num_to_move = max(500, int(total/11))

    files_to_move = pkl_files[-num_to_move:]

    print(f"Total {total} .pkl files, moving the last {num_to_move} to '{val_dir}' directory.")


    for file_path in files_to_move:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(val_dir, file_name)
        shutil.move(file_path, dest_path)

    print("File moving completed.")