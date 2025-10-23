from face_blurrer import FaceBlurrer


model_path = './models/blaze_face_short_range.tflite'
face_blurrer = FaceBlurrer(model_path=model_path)
face_blurrer.live_face_blur()
