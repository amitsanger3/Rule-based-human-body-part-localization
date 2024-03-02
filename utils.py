from frame.fashion_frame import *
from Harcascade_Face_Detection.Faces import *
from Harcascade_Face_Detection.cnn import *
from intensity_person_detection.person import *
from yolo_utils.face_detection import *
from yolo_utils.person_detection import *


class ImageFashionFrameDetection(object):

    def __init__(self, image_path, face_cascade, yolo_face_model_path, output_dir,
                 cnn_model_path, device,
                 use_yolo=False,
                 save_detections=False,
                 display=False):
        self.image_path = image_path
        self.yolo_face_model_path = yolo_face_model_path
        self.face_detector = Faces(face_cascade)
        self.cnn_model = NodeCNN()
        self.output_dir = output_dir
        self.use_yolo = use_yolo
        self.save_detections = save_detections
        self.display = display
        self.check_and_create_dir(self.output_dir)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        self.read_image()

    def read_image(self):
        if self.image:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        else:
            self.image = None

    @staticmethod
    def check_and_create_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def haarcascade_face_detection(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        face_co_ordinates = self.face_detector.face_co_ordinates(gray)

        if len(co_ordinates) > 0:
            return face_co_ordinates
        return None

    def yolo_face_detect(self):
        return faces(model_path=self.yolo_face_model_path, image_path=self.image_path)

    def intensity_person_detect(self, threshold=10):
        detector = IntensityPersonDetect(self.image)
        return detector.detect(threshold=threshold)

    def yolo_person_detect(self):
        detections = predict(image_path=self.image_path)
        face_co_ordinates = []
        for detection in detections:
            x1,y1, x2,y2 = detection.xyxy
            face_co_ordinates.append([x1,y1, x2-x1, y2-y1])
        if len(face_co_ordinates) > 0:
            return face_co_ordinates
        return None

    def frames_detection(self):
        if self.image is None:
            print("No image found to detect Fashion Frames...")
        else:
            does_get_any_face = False
            if self.use_yolo:
                face_coordinates = self.yolo_face_detect()
                person_coordinates = self.yolo_person_detect()
            else:
                face_coordinates = self.haarcascade_face_detection()
                person_coordinates = self.intensity_person_detect()

            if face_coordinates:
                for n, co_ordinate in enumerate(face_coordinates):
                    x, y, w, h = co_ordinate
                    try:
                        crop_image = self.face_detector.crop_face(image, co_ordinate)
                        person_op_dir = None
                        if self.save_detections:
                            person_op_dir = os.path.join(os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.image_path))[0]),
                                                       f"person_{n}")
                            self.check_and_create_dir(person_op_dir)
                        name = os.path.join(person_op_dir, 'face.jpg')
                        cv2.imwrite(name, crop_image)

                    except:
                        print(traceback.print_exc())
                    try:
                        x_ = cv2.resize(cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY), (2 * 16, 32)) / 255
                        x_ = x_.astype(np.float32)
                        x_ = torch.tensor(x_)
                        x_ = x_.view((1, 1, 32, 16 * 2))
                        pred = self.cnn_model(x_)

                        pred = int(pred.argmax().item())

                        if pred == 1:
                            #                                     print("true face detected ----------- ")
                            does_get_any_face = True

                            # Now detect person in it.
                            for i in range(0,len(person_coordinates), 2):
                                try:
                                    (px1, py1), (px2, py2) = person_coordinates[i], person_coordinates[i+1]
                                    if px1 < x < px2 and py1 < y < py2:
                                        fashion_frames_detector = FashionFrameDetection(
                                            image=self.image,
                                            face_coordinates=co_ordinate,
                                            person_frame_coordinates=((px1, py1), (px2, py2)),
                                            colours=colours,
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
        if self.save_detections:
            cv2.imwrite(os.path.join(self.output_dir, os.path.basename(self.image_path)), self.image)
        return None


class VideoFashionFrameDetection(object):

    def __init__(self, video_path, face_cascade, yolo_face_model_path, output_dir,
                 cnn_model_path, device,
                 use_yolo=False,
                 save_detections=False,
                 display=False):

        self.video = video_path

        self.op_img_dir = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.video))[0])
        self.check_and_create_dir(self.op_img_dir)

        self.image_frame_detector = ImageFashionFrameDetection(image_path=None, face_cascade=face_cascade,
                                                          yolo_face_model_path=yolo_face_model_path,
                                                          output_dir=None, cnn_model_path=cnn_model_path,
                                                          device=device, use_yolo=use_yolo,
                                                          save_detections=save_detections,
                                                          display=display)

    @staticmethod
    def check_and_create_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def video_processor(self):

        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            print("Error opening video file")
            return None
        print(self.video_num, 'Video Running -->', self.video)

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the frame size of the video
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Create a VideoWriter object to write the output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(self.output_dir, os.path.basename(self.video)), fourcc, fps, frame_size)

        img_num = 1
        while True:
            success, image = cap.read()
            try:
                image = np.uint8(image)
                image_name = os.path.join(self.op_img_dir, f"frame_{img_num}.jpg")
                cv2.imwrite(image_name, image)

                self.image_frame_detector.image_path = image_name
                self.image_frame_detector.read_image()
                self.image_frame_detector.frames_detection()
                out.write(self.image_frame_detector.image)
            except:
                break

        cap.release()
        out.release()
        return None



