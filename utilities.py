import pandas as pd
import shutil
import time

from frame.fashion_frame import *
from Harcascade_Face_Detection.Faces import *
from Harcascade_Face_Detection.cnn import *
from intensity_person_detection.person import *
from yolo_utils.face_detection import *
from yolo_utils.person_detection import *

from config import *


class ImageFashionFrameDetection(object):

    def __init__(self, image_path, face_cascade, yolo_face_model_path, output_dir,
                 cnn_model_path, device,
                 use_yolo=False,
                 use_fcn=False,
                 save_detections=False,
                 display=False):
        self.image_path = image_path
        self.yolo_face_model_path = yolo_face_model_path
        self.face_detector = Faces(face_cascade)
        self.cnn_model = NodeCNN()
        self.output_dir = output_dir
        self.use_yolo = use_yolo
        self.use_fcn = use_fcn
        self.is_video = False
        self.save_detections = save_detections
        self.display = display
        self.check_and_create_dir(self.output_dir)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))

        self.check_and_create_dir(EVAL_DIR)
        self.img_df_path = os.path.join(EVAL_DIR, IMAGES_EVAL_FILE)
        if os.path.exists(self.img_df_path):
            self.img_dict = pd.read_csv(self.img_df_path).to_dict('list')
        else:
            self.img_dict = {
                "Image_path": [], "Haarcascade_face":[],
                "Yolo_face": [], "FCN":[], "Face_after_fcn": [],
                "Intensity_person": [], "Yolo_Person": [],
                "Person_Coordinates": []
            }

        self.read_image()

    def read_image(self):
        # print("4------------", self.image_path)
        if self.image_path is not None:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        else:
            # print("4------------", self.image_path)
            self.image = None

    @staticmethod
    def check_and_create_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def haarcascade_face_detection(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        face_co_ordinates = self.face_detector.face_co_ordinates(gray)

        if len(face_co_ordinates) > 0:
            return face_co_ordinates
        return None

    def yolo_face_detect(self):
        return faces(model_path=self.yolo_face_model_path, image_path=self.image_path)

    def intensity_person_detect(self, threshold=10):
        detector = IntensityPersonDetect(self.image)
        return detector.detect(threshold=threshold)

    def yolo_person_detect(self):
        detections = predict(image_path=self.image_path, weights=yolo_person_detection_model_path, device=device)
        # print("Detections:" , detections)
        face_co_ordinates = []
        for detection in detections:
            # print("Yolo person detect: ", detection)
            cords = detection.xyxy[0]
            cords = list(map(int, cords))
            face_co_ordinates.append(cords)
        if len(face_co_ordinates) > 0:
            return face_co_ordinates
        return None

    def frames_detection(self):
        if self.image is None and self.image_path not in self.img_dict["Image_path"]:
            self.img_dict["Image_path"].append(self.image_path)
            self.img_dict["Haarcascade_face"].append(None)
            self.img_dict["Yolo_face"].append(None)
            self.img_dict["FCN"].append(None)
            self.img_dict["Face_after_fcn"].append(None)
            self.img_dict["Intensity_person"].append(None)
            self.img_dict["Yolo_Person"].append(None)
            self.img_dict["Person_Coordinates"].append(None)
            print("No image found to detect Fashion Frames...")
        elif self.image is None and self.image_path in self.img_dict["Image_path"]:
            pass
        elif self.image is not None and self.image_path not in self.img_dict["Image_path"]:
            self.img_dict["Image_path"].append(self.image_path)
            print("1 -----------============", self.use_yolo, self.use_fcn)
            does_get_any_face = False
            if self.use_yolo:
                try:
                    face_coordinates = self.yolo_face_detect()
                    start = time.time()
                    person_coordinates = self.yolo_person_detect()
                    print("Yolo Person time: \t", time.time()-start)
                except:
                    self.img_dict["Haarcascade_face"].append(None)
                    self.img_dict["Yolo_face"].append(None)
                    self.img_dict["FCN"].append(None)
                    self.img_dict["Face_after_fcn"].append(None)
                    self.img_dict["Intensity_person"].append(None)
                    self.img_dict["Yolo_Person"].append(None)
                    self.img_dict["Person_Coordinates"].append(None)
                    img_dict_df = pd.DataFrame(self.img_dict, index=np.arange(len(self.img_dict["Image_path"])))
                    img_dict_df.to_csv(os.path.join(EVAL_DIR, IMAGES_EVAL_FILE), index=False)
                    print("No image found to detect Fashion Frames...")
                    return None
                self.img_dict["Haarcascade_face"].append(None)
                self.img_dict["Intensity_person"].append(None)
                self.img_dict["Yolo_face"].append(0 if face_coordinates is None else len(face_coordinates))
                self.img_dict["Yolo_Person"].append(0 if person_coordinates is None else len(person_coordinates))
            else:
                try:
                    face_coordinates = self.haarcascade_face_detection()
                    start = time.time()
                    person_coordinates = self.intensity_person_detect()
                    print("Intensity Person time: \t", time.time() - start)
                except:
                    self.img_dict["Haarcascade_face"].append(None)
                    self.img_dict["Yolo_face"].append(None)
                    self.img_dict["FCN"].append(None)
                    self.img_dict["Face_after_fcn"].append(None)
                    self.img_dict["Intensity_person"].append(None)
                    self.img_dict["Yolo_Person"].append(None)
                    self.img_dict["Person_Coordinates"].append(None)
                    img_dict_df = pd.DataFrame(self.img_dict, index=np.arange(len(self.img_dict["Image_path"])))
                    img_dict_df.to_csv(os.path.join(EVAL_DIR, IMAGES_EVAL_FILE), index=False)
                    print("No image found to detect Fashion Frames...")
                    return None
                self.img_dict["Haarcascade_face"].append(0 if face_coordinates is None else len(face_coordinates))
                self.img_dict["Intensity_person"].append(0 if person_coordinates is None else len(person_coordinates))
                self.img_dict["Yolo_face"].append(None)
                self.img_dict["Yolo_Person"].append(None)

            face_after_fcn = 0
            image_person_coordinate_collection = []
            self.img_dict["FCN"].append(1 if self.use_fcn else 0)
            if face_coordinates is not None:
                for n, co_ordinate in enumerate(face_coordinates):
                    if self.use_yolo:
                        x, y, x_c2, y_c2 = co_ordinate[0]
                        co_ordinate = x, y, x_c2-x, y_c2-y
                        co_ordinate = list(map(int, co_ordinate))
                        print("co ordinates: ", co_ordinate)
                    x, y, w, h = co_ordinate
                    person_op_dir = os.path.join(
                        os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.image_path))[0]),
                        f"person_{n}")
                    self.check_and_create_dir(person_op_dir)

                    try:
                        crop_image = self.face_detector.crop_face(self.image, co_ordinate)
                        # person_op_dir = None
                        if self.save_detections:

                            name = os.path.join(person_op_dir, 'face.jpg')
                            cv2.imwrite(name, crop_image)

                    except:
                        crop_image = None
                        print(traceback.print_exc())
                    try:
                        if crop_image is not None:
                            if self.use_fcn:
                                x_ = cv2.resize(cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY), (2 * 16, 32)) / 255
                                x_ = x_.astype(np.float32)
                                x_ = torch.tensor(x_)
                                x_ = x_.view((1, 1, 32, 16 * 2))
                                pred = self.cnn_model(x_)

                                pred = int(pred.argmax().item())
                            else:
                                pred = 1

                            if pred == 1:

                                try:
                                    if not self.use_yolo:
                                        intensity_image = cv2.resize(cv2.imread(os.path.join(TEMP_DIR, TEMP_INTENSITY_IMG)),
                                                                     (self.image.shape[1], self.image.shape[0]))
                                        x_intensity_image = cv2.resize(cv2.imread(os.path.join(TEMP_DIR, TEMP_X_AXIS_IMG)),
                                                                       (self.image.shape[1], self.image.shape[0]))
                                        y_intensity_image = cv2.resize(cv2.imread(os.path.join(TEMP_DIR, TEMP_Y_AXIS_IMG)),
                                                                       (self.image.shape[1], self.image.shape[0]))
                                        cv2.imwrite(os.path.join(person_op_dir, "Person_intensity_evaluation.jpg"),
                                                    np.vstack(
                                                        (np.hstack((self.image, y_intensity_image)),
                                                         np.hstack((x_intensity_image, intensity_image)))
                                                    ))
                                except:
                                    print(traceback.print_exc())
                                    pass

                                face_after_fcn += 1
                                does_get_any_face = True

                                # Now detect person in it.
                                person_coordinate_collection = []
                                if self.use_yolo:
                                    for i in range(len(person_coordinates)):
                                        try:
                                            px1, py1, px2, py2 = person_coordinates[i]
                                            if px1 < x < px2 and py1 < y < py2:
                                                person_coordinate_collection.append(((px1, py1), (px2, py2)))
                                        except:
                                            print(traceback.print_exc())
                                            continue
                                else:
                                    for i in range(0,len(person_coordinates), 2):
                                        try:
                                            (px1, py1), (px2, py2) = person_coordinates[i], person_coordinates[i+1]
                                            if px1 < x < px2 and py1 < y < py2:
                                                person_coordinate_collection.append(((px1, py1), (px2, py2)))
                                        except:
                                            print(traceback.print_exc())
                                            continue
                                # print("2----------", person_coordinate_collection)
                                image_person_coordinate_collection.append(person_coordinate_collection)
                                for valid_person_coordinates in person_coordinate_collection:
                                    # print("valid person coordinate: --", valid_person_coordinates)
                                    if not self.save_detections:
                                        person_op_dir = None
                                    try:
                                        fashion_frames_detector = FashionFrameDetection(
                                            image=self.image,
                                            face_coordinates=co_ordinate,
                                            person_frame_coordinates=valid_person_coordinates,
                                            colours=colors,
                                            output_dir=person_op_dir,
                                            display=self.display)
                                        self.image = fashion_frames_detector.frame_detect()
                                    except:
                                        print(traceback.print_exc())
                                        continue
                    except:
                        print(traceback.print_exc())
            if not does_get_any_face:
                print("No face found in this image.")
            if self.use_fcn:
                self.img_dict["Face_after_fcn"].append(face_after_fcn)
            else:
                self.img_dict["Face_after_fcn"].append(None)
            self.img_dict["Person_Coordinates"].append(str(image_person_coordinate_collection))
        for k in self.img_dict.keys():
            print(k, len(self.img_dict[k]))
        img_dict_df = pd.DataFrame(self.img_dict, index=np.arange(len(self.img_dict["Image_path"])))
        img_dict_df.to_csv(os.path.join(EVAL_DIR, IMAGES_EVAL_FILE), index=False)
        # print("Image Evaluations saved to csv", os.path.join(EVAL_DIR, IMAGES_EVAL_FILE), self.img_dict)
        if self.save_detections:
            cv2.imwrite(os.path.join(self.output_dir, os.path.basename(self.image_path)), self.image)
        return None


class VideoFashionFrameDetection(object):

    def __init__(self, video_path, face_cascade, yolo_face_model_path, output_dir,
                 cnn_model_path, device,
                 use_yolo=False,
                 use_fcn=False,
                 save_detections=False,
                 display=False):

        self.video = video_path
        self.output_dir = output_dir
        self.op_img_dir = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.video))[0])
        self.raw = os.path.join(self.op_img_dir, RAW_DIR_NAME)
        self.check_and_create_dir(self.op_img_dir)
        self.check_and_create_dir(self.raw)
        self.image_frame_detector = ImageFashionFrameDetection(image_path=None, face_cascade=face_cascade,
                                                          yolo_face_model_path=yolo_face_model_path,
                                                          output_dir=self.op_img_dir, cnn_model_path=cnn_model_path,
                                                          device=device, use_yolo=use_yolo, use_fcn=use_fcn,
                                                          save_detections=save_detections,
                                                          display=display)
        self.check_and_create_dir(EVAL_DIR)
        self.vid_df_path = os.path.join(EVAL_DIR, VIDEO_EVAL_FILE)
        if os.path.exists(self.vid_df_path):
            self.vid_dict = pd.read_csv(self.vid_df_path).to_dict('list')
        else:
            self.vid_dict = {
                "Video_path": []
            }

    @staticmethod
    def check_and_create_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def video_processor(self):
        if self.video not in self.vid_dict["Video_path"]:
            cap = cv2.VideoCapture(self.video)
            if not cap.isOpened():
                print("Error opening video file")
                return None
            print('Video Running -->', self.video)

            # Get the frame rate of the video
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Get the frame size of the video
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            # Create a VideoWriter object to write the output
            if os.path.splitext(os.path.basename(self.video))[-1] == ".mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif os.path.splitext(os.path.basename(self.video))[-1] == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                print("Not a valid video format.")
                return None
            out = cv2.VideoWriter(os.path.join(self.output_dir, os.path.basename(self.video)), fourcc, fps, frame_size)

            img_num = 1
            while True:
                success, image = cap.read()
                try:
                    image = np.uint8(image)
                    image_name = os.path.join(self.raw, f"frame_{img_num}.jpg")
                    cv2.imwrite(image_name, image)
                    self.image_frame_detector.is_video = True
                    self.image_frame_detector.image_path = image_name
                    self.image_frame_detector.read_image()
                    self.image_frame_detector.frames_detection()
                    out.write(self.image_frame_detector.image)
                    img_num += 1
                except:
                    print(traceback.print_exc())
                    break

            cap.release()
            out.release()

            self.vid_dict['Video_path'].append(self.video)
            self.vid_dict = pd.DataFrame(self.vid_dict, index=np.arange(len(self.vid_dict['Video_path'])))
            self.vid_dict.to_csv(os.path.join(EVAL_DIR, VIDEO_EVAL_FILE), index=False)
            print("Video Evaluations saved to csv", os.path.join(EVAL_DIR, VIDEO_EVAL_FILE))
        shutil.rmtree(self.raw, ignore_errors=True)
        return None



