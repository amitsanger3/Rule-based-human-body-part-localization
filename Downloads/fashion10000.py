import pandas as pd
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
save_dir = r"F:\Documents\PROJECT\GARMENTS\CROQUIE_DC\DownloadDataset\Fashion1000"
os.makedirs(save_dir, exist_ok=True)

json_file="fashion100000.json"
if os.path.exists(json_file):
    with open(json_file, 'r') as op:
        j_data = json.load(op)
        op.close()
else:
    j_data = {"urls": []}


df = pd.read_csv('ImageUrls.csv')
df = df.sample(10000)
urls = df['url']
n=0
for i, url in enumerate(list(urls)):
    if url not in j_data["urls"]:
        try:
            res = requests.get(url, timeout=5)
            if res.status_code==200:
                with open(os.path.join(save_dir, f'{i}.jpg'), 'wb') as fl:
                    fl.write(res.content)
                    fl.close()
                n+=1
        except Exception as e:
            print("Error:", e)
        with open(json_file, 'w+') as op:
            json.dump(j_data, op)
            op.close()
        time.sleep(1)

