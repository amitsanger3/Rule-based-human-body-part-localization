import requests
from PIL import Image
import os
import io
from tqdm import tqdm
import json
import time

# Define minimum image size (pixels)
min_width = 300
min_height = 200

# Set directory to save downloaded images (create if it doesn't exist)
save_dir = r"F:\Documents\PROJECT\GARMENTS\CROQUIE_DC\DownloadDataset\Darn"
os.makedirs(save_dir, exist_ok=True)

json_file="darn.json"
if os.path.exists(json_file):
    with open(json_file, 'r') as op:
        j_data = json.load(op)
        op.close()
else:
    j_data = {"urls": []}


# Open the file containing image names and URLs
with open("darn_url.txt", "r") as f:
    for line in tqdm(f):
        # Split line into image name and URL
        image_name, url = line.strip().split(" ")
        # url = url.strip()
        image_name = image_name.strip()
        if url not in j_data["urls"]:
            # Try downloading the image
            try:
                response = requests.get(url,
                                        headers={'User-Agent':'Mozilla5.0(Google spider)'})
                print(response.status_code)
                # Check image size before saving
                with Image.open(io.BytesIO(response.content)) as img:
                    width, height = img.size
                    if width >= min_width and height >= min_height:
                        # Save the image
                        save_path = os.path.join(save_dir, image_name)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(save_path, "wb") as f:
                            f.write(response.content)
                            f.close()
                        # print(f"Downloaded and saved: {image_name}")
                    else:
                        # print(f"Image {image_name} is too small ({width}x{height}), skipping.")
                        pass

            except Exception as e:
                print(f"Error downloading {url}: {e}")
                pass
            time.sleep(1)
            j_data["urls"].append(url)

            with open(json_file, 'w+') as op:
                json.dump(j_data, op)
                op.close()
