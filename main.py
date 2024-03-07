from config import *
from utilities import *

import os, sys


from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


sys.path.append(os.path.dirname('./yolov9/'))


def dataset_processor(dataset_path, output_path):
    harcascade_face_classifier = cv2.CascadeClassifier(os.path.join(harcascade_classifier_path, os.listdir(harcascade_classifier_path)[0]))
    if os.path.isfile(dataset_path):
        if os.path.splitext(os.path.basename(dataset_path))[-1] in allowed_files['video']:
            frame_detector = VideoFashionFrameDetection(video_path=dataset_path,
                                                        face_cascade=harcascade_face_classifier,
                                                        yolo_face_model_path=yolo_face_detection_model_path,
                                                        output_dir=output_path,
                                                        cnn_model_path=cnn_model_path, device=device,
                                                        use_yolo=True,
                                                        save_detections=True,
                                                        display=True)
            frame_detector.video_processor()

        elif os.path.splitext(os.path.basename(dataset_path))[-1] in allowed_files['image']:
            frame_detector = ImageFashionFrameDetection(image_path=dataset_path,
                                                        face_cascade=harcascade_face_classifier,
                                                        yolo_face_model_path=yolo_face_detection_model_path,
                                                        output_dir=output_path,
                                                        cnn_model_path=cnn_model_path, device=device,
                                                        use_yolo=True,
                                                        save_detections=True,
                                                        display=True)
            frame_detector.frames_detection()
        else:
            print(dataset_path, "Not a valid filetype. Please Check")
    else:
        dataset_content = os.listdir(dataset_path)
        for content in dataset_content:
            dataset_processor(dataset_path=os.path.join(dataset_path, content), output_path=os.path.join(output_path, content))
        
        
if __name__ == "__main__":
    sys.path.append(os.path.dirname('./yolov9/'))
    # dataset_processor(dataset_path="F:\Documents\PROJECT\GARMENTS\CROQUIE_DC\Rule-based-human-body-part-localization\Dataset\Fashion-Dataset-Images-Western-Dress-master\WesternDress_Images",
    #                   output_path="F:\Documents\PROJECT\GARMENTS\CROQUIE_DC\Rule-based-human-body-part-localization\Output")
    dataset_processor(
        dataset_path=r"F:\Documents\PROJECT\GARMENTS\CROQUIE_DC\Rule-based-human-body-part-localization\Dataset\Vids",
        output_path=r"F:\Documents\PROJECT\GARMENTS\CROQUIE_DC\Rule-based-human-body-part-localization\Output")