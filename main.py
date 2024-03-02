from config import *
from utils import *


def dataset_processor(dataset_path, output_path):
    if os.path.isfile(dataset_path):
        if os.path.splitext(os.path.basename(dataset_path))[-1] in allowed_files['video']:
            frame_detector = VideoFashionFrameDetection(video_path=dataset_path,
                                                        face_cascade=harcascade_classifier_path,
                                                        yolo_face_model_path=yolo_face_detection_model_path,
                                                        output_dir=output_path,
                                                        cnn_model_path=cnn_model_path, device=device,
                                                        use_yolo=False,
                                                        save_detections=False,
                                                        display=False)
            frame_detector.frames_detection()

        elif os.path.splitext(os.path.basename(dataset_path))[-1] in allowed_files['image']:
            frame_detector = ImageFashionFrameDetection(image_path=dataset_path,
                                                        face_cascade=harcascade_classifier_path,
                                                        yolo_face_model_path=yolo_face_detection_model_path,
                                                        output_dir=output_path,
                                                        cnn_model_path=cnn_model_path, device=device,
                                                        use_yolo=False,
                                                        save_detections=False,
                                                        display=False)
            frame_detector.video_processor()
        else:
            print(dataset_path, "Not a valid filetype. Please Check")
    dataset_content = os.listdir(dataset_path)
    for content in dataset_content:
        dataset_processor(dataset_path=os.path.join(dataset_path, content), output_path=os.path.join(output_path, content))
        
        
if __name__ == "__main__":
    dataset_processor(dataset_path="", output_path="")