import torch

harcascade_classifier_path = r'F:\Documents\PROJECT\Face\HAARCASCADE'
model_path = r'F:\Documents\PROJECT\Face\FACE_CLASSIFIER_MODEL_2'
yolo_face_detection_model_path = ""

cnn_model_path = os.path.join(model_path, "FRS_CLASSIFIER_Model_50.net")
device=torch.device("cpu")

colors = {
    "face": "#ff3333",
    "neck": "#ff9933",
    "deep-neck": "#ffff33",
    "upper-half-torso": "#33ff33",
    "lower-half-torso": "#3333ff",
    "arm": "#33ffff",
    "hand": "#ff33ff",
    "legs": "#bfff00",
    "feet": "#4d4d4d",
    "fullness": "#aa00ff",
    "list": ["#FF0000", "#00FF00", "#0000FF", "#FFD700", "#800080", "#008080", "#FFA500", "#00FFFF", "#FF00FF",
             "#FFFF00", "#008000", "#800000", "#808080", "#00FF80", "#FF8080", "#8000FF", "#FF6347", "#40E0D0",
             "#8A2BE2", "#FA8072"]
}

allowed_files = {
    "video":['.mp4', '.avi'],
    "image":['jpg', 'png']
}
