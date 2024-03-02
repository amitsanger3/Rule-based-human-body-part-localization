from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image



def faces(model_path, image_path):
    model = YOLO(model_path)

    # inference
    output = model(Image.open(image_path))
    results = Detections.from_ultralytics(output[0])
    return results
