import os, shutil
from glob import glob
import random

def sample_images(src_dir, dest_dir, samples_per_class=100):
    os.makedirs(dest_dir, exist_ok=True)
    for label in os.listdir(src_dir):
        class_src = os.path.join(src_dir, label)
        class_dest = os.path.join(dest_dir, label)
        os.makedirs(class_dest, exist_ok=True)

        images = glob(os.path.join(class_src, '*'))
        sample = random.sample(images, min(samples_per_class, len(images)))

        for img in sample:
            shutil.copy(img, class_dest)

# Paths to your full original dataset
original_train = 'data/data/train'
original_val = 'data/data/val'
original_test = 'data/data/test'

# Paths to where sampled images will go
sampled_train = 'sampled_data/train'
sampled_val = 'sampled_data/val'
sampled_test = 'sampled_data/test'

# Run sampling
sample_images(original_train, sampled_train)
sample_images(original_val, sampled_val)
sample_images(original_test, sampled_test)

print("✅ Sampling complete!")
