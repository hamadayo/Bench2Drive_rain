from PIL import Image

image_path = "/home/yoshi-22/Bench2Drive/labels/center_rain.png"

with Image.open(image_path) as img:
    width, height = img.size
    print(f"Width: {width}, Height: {height}")