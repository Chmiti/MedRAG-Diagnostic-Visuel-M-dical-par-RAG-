# convert_test_dcm_to_jpg.py
import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_FOLDER = "data/stage_2_test_images"
OUTPUT_FOLDER = "data/test_jpg"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in tqdm(os.listdir(INPUT_FOLDER)):
    if not filename.endswith(".dcm"):
        continue
    path = os.path.join(INPUT_FOLDER, filename)
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    img = Image.fromarray(img).convert("RGB")
    img.save(os.path.join(OUTPUT_FOLDER, filename.replace(".dcm", ".jpg")))
