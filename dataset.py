import os
import shutil
import random

input_dir = "Dataset2"
output_dir = "dataset_split_2"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

splits = ['train', 'val', 'test']

for split in splits:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

classes = [
    c for c in os.listdir(input_dir)
    if os.path.isdir(os.path.join(input_dir, c))
]

print("Classes found:", classes)

for cls in classes:
    class_path = os.path.join(input_dir, cls)

    images = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ]

    print(f"Processing {cls} - {len(images)} images")

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    for split, files in {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }.items():

        split_class_dir = os.path.join(output_dir, split, cls)
        os.makedirs(split_class_dir, exist_ok=True)

        for f in files:
            src = os.path.join(class_path, f)
            dst = os.path.join(split_class_dir, f)
            shutil.copy2(src, dst)

print("Done splitting dataset")
print("OUTPUT PATH:", os.path.abspath(output_dir))