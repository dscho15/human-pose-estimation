import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2
import mediapy

from tqdm import tqdm

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
      
    pose_landmarks = pose_landmarks_list[idx]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  
  return annotated_image
    
P_CHECKPT = "checkpoints/pose_landmarker.task"
P_IMG = "imgs/image.jpg"

# load video
video = mediapy.read_video("videoes/1f2a37641863491588b98e1dfa4de062.mp4")

base_options = python.BaseOptions(model_asset_path=P_CHECKPT)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.PoseLandmarkerOptions.RunningMode.ASYNC,
    output_segmentation_masks=True) 
detector = vision.PoseLandmarker.create_from_options(options)

# frame by frame predict and draw
annots = []
for f in tqdm(video[200:250]):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(f))
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    annots.append(annotated_image)

# save video 
mediapy.write_video(f"results/1f2a37641863491588b98e1dfa4de062.mp4", annots, fps=30)