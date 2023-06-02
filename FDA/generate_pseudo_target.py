from FDA import apply_fda
from day_to_night import simulate_night
from tqdm import tqdm
import sys
import os
import random
from glob import glob

source_dir = sys.argv[1]
target_dir = sys.argv[2]
output_dir = sys.argv[3]

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

source_images_paths = sorted(glob(f"{source_dir}/**/*.jpg", recursive=True))
target_images_paths = glob(f"{target_dir}/*.jpg", recursive=True)

for source_filename in tqdm(source_images_paths):
    source_internal_dir=source_filename.split("/")[-2]
    source_image_name=source_filename.split("/")[-1]
    source_night_image=source_image_name.replace('@', 'NIGHT@', 1)
    index=int(random.uniform(0,105))
    image_fda_applied=apply_fda(source_filename,target_images_paths[index])
    image_simulate_night_applied=simulate_night(image_fda_applied)
    if not os.path.isdir(os.path.join(output_dir, source_internal_dir)):
        os.makedirs(os.path.join(output_dir, source_internal_dir))
    image_simulate_night_applied.save(os.path.join(output_dir, source_internal_dir, source_night_image))