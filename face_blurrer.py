import cv2
import mediapipe as mp
import os

class FaceBlurrer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.live_detection_result = None

    def process_image(self, image, detector):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        face_detection_result = detector.detect(mp_image)
        for detection in face_detection_result.detections:
            bbox = detection.bounding_box
            x1, y1, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            image[y1:y1 + height, x1:x1 + width, :] = cv2.blur(image[y1:y1 + height, x1:x1 + width, :] , (200, 200))
        return image
    
    def process_video(self, image, timestamp, detector):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        face_detection_result = detector.detect_for_video(mp_image, int(100 * timestamp))
        for detection in face_detection_result.detections:
            bbox = detection.bounding_box
            x1, y1, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            image[y1:y1 + height, x1:x1 + width, :] = cv2.blur(image[y1:y1 + height, x1:x1 + width, :] , (200, 200))
        return image
    
    def process_live(self, result, output_image, timestamp):
        self.live_detection_result = result
    
    def image_face_blur(self, image_path):
        image = cv2.imread(image_path)
        options = self.get_image_face_detector_options(model_path=self.model_path)
        FaceDetector = mp.tasks.vision.FaceDetector
        with FaceDetector.create_from_options(options) as face_detection:
            image_blurred = self.process_image(image, face_detection)
        image_name = image_path.split('/')[-1]
        self.save_image(image_blurred, image_name)
        return image_blurred
    
    def video_face_blur(self, video_path):
        cap = cv2.VideoCapture(video_path)
        options = self.get_video_face_detector_options(model_path=self.model_path)
        FaceDetector = mp.tasks.vision.FaceDetector
        with FaceDetector.create_from_options(options) as face_detection:
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame = self.process_video(frame, timestamp_ms, face_detection)
                frames.append(frame)
            cap.release()
        video_file_name = video_path.split('/')[-1]
        self.save_video(frames, video_file_name=video_file_name)

    def live_face_blur(self):
        cap = cv2.VideoCapture(0)
        options = self.get_livestream_face_detector_options(model_path=self.model_path)
        FaceDetector = mp.tasks.vision.FaceDetector
        with FaceDetector.create_from_options(options) as face_detection:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                face_detection.detect_async(mp_frame, int(100 * timestamp_ms))
                if self.live_detection_result:
                    for detection in self.live_detection_result:
                        bbox = self.live_detection_result.bounding_box
                        x1, y1, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                        frame[y1:y1 + height, x1:x1 + width, :] = cv2.blur(frame[y1:y1 + height, x1:x1 + width, :] , (200, 200))
                cv2.imshow('Video', frame)
                cv2.waitKey(20)
        cap.release()
    
    def save_image(self, image, image_name):
        if not os.path.exists('./outputs/images'):
            os.makedirs('./outputs/images', exist_ok=True)
        cv2.imwrite(os.path.join('./outputs/images', image_name), image)

    def save_video(self, video_frames, video_file_name):
        if not os.path.exists('./outputs/video'):
            os.makedirs('./outputs/video', exist_ok=True)
        video_file_no_extension = video_file_name.split('.')[0]
        video_path = video_file_no_extension + '.' + 'avi'
        output_video = cv2.VideoWriter(os.path.join('./outputs/video', video_path),
                                       cv2.VideoWriter.fourcc(*'XVID'),
                                       25,
                                       (video_frames[0].shape[1], video_frames[0].shape[0]))
        for video_frame in video_frames:
            output_video.write(video_frame)
        output_video.release()

    def get_image_face_detector_options(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        return options
    
    def get_video_face_detector_options(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        return options
    
    def get_livestream_face_detector_options(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.process_live
        )
        return options