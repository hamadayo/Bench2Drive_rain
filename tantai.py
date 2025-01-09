
import csv
from raindrop.dropgenerator import generateDrops, generate_label
from raindrop.config import cfg
from PIL import Image
from raindrop.raindrop import Raindrop
import os
import numpy as np

step = 0
cam = "front"
image_path = "/home/yoshi-22/Bench2Drive/eval_v1/1711_all/RouteScenario_0_rep0_Town12_ParkingCutIn_1_None_12_20_13_01_45/rgb_front/0000.png"
if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
frame = np.array(Image.open(image_path))
pil_image = Image.fromarray(frame)

# 保存先ディレクトリを指定
# label_save_dir = "/home/yoshi-22/Bench2Drive/reliability/ri_label"
# os.makedirs(label_save_dir, exist_ok=True)

out_folder = "/home/yoshi-22/Bench2Drive/raindrop"
mask_folder = "/home/yoshi-22/Bench2Drive/mask"

os.makedirs(out_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# 保存先ディレクトリを指定
csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops.csv"
raindrops = []
with open(csv_file_path, "r") as csv_file:

    reader = csv.DictReader(csv_file)
    for row in reader:
        key = int(row["Key"])
        center_x = int(row["CenterX"])
        center_y = int(row["CenterY"])
        radius = int(row["Radius"])
        shape = int(row["Shape"])
        raindrop = Raindrop(key, (center_x, center_y), radius, shape=shape)
        raindrops.append(raindrop)

# 雨滴を画像に追加
output_image, output_label, mask = generateDrops(pil_image, cfg, raindrops)
file_name = f"raindrop_{step}_{cam}.png"


# 画像とマスクの保存
save_path = os.path.join(out_folder, file_name)
mask_path = os.path.join(mask_folder, file_name)

output_image.save(save_path)
mask.save(mask_path)