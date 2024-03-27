import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2
import mediapy

from tqdm import tqdm

lookup_table = {
  # "NOSE": 0,
  # "LEFT_EYE_INNER": 1,
  # "LEFT_EYE": 2,
  # "LEFT_EYE_OUTER": 3,
  # "RIGHT_EYE_INNER": 4,
  # "RIGHT_EYE": 5,
  # "RIGHT_EYE_OUTER": 6,
  # "LEFT_EAR": 7,
  # "RIGHT_EAR": 8,
  # "MOUTH_LEFT": 9,
  # "MOUTH_RIGHT": 10,
  "LEFT_SHOULDER": 11,
  "RIGHT_SHOULDER": 12,
  "LEFT_ELBOW": 13,
  "RIGHT_ELBOW": 14,
  "LEFT_WRIST": 15,
  "RIGHT_WRIST": 16,
  "LEFT_PINKY": 17,
  "RIGHT_PINKY": 18,
  "LEFT_INDEX": 19,
  "RIGHT_INDEX": 20,
  "LEFT_THUMB": 21,
  "RIGHT_THUMB": 22,
  # "LEFT_HIP": 23,
  # "RIGHT_HIP": 24,
  # "LEFT_KNEE": 25,
  # "RIGHT_KNEE": 26,
  # "LEFT_ANKLE": 27,
  # "RIGHT_ANKLE": 28,
  # "LEFT_HEEL": 29,
  # "RIGHT_HEEL": 30,
  # "LEFT_FOOT_INDEX": 31,
  # "RIGHT_FOOT_INDEX": 32
}



def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result, segm_mask: bool = False):
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
    
  # Draw the segmentation mask on the image.
  if detection_result.segmentation_masks and segm_mask:
    
    segm_mask = detection_result.segmentation_masks[0]
    segm_mask = segm_mask.numpy_view() * 255.0
    segm_mask = segm_mask.astype(np.uint8)
    segm_mask = cv2.cvtColor(segm_mask, cv2.COLOR_GRAY2BGR)
    annotated_image = cv2.addWeighted(annotated_image, 0.5, segm_mask, 0.5, 0)
  
  return annotated_image

def extract_landmark_trajectories(detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  
  traj = []
  
  # Trajectory for elbow, wrist and shoulder
  # mp_pose = mp.solutions.pose
  mp_keys = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
  # poses

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]    
    for i in mp_keys:
      landmark = pose_landmarks[i]
      traj.append([landmark.x, landmark.y, landmark.z])
  return traj
    
P_CHECKPT = "checkpoints/pose_landmarker.task"
P_IMG = "imgs/image.jpg"

# load video
video = mediapy.read_video("videoes/1f2a37641863491588b98e1dfa4de062.mp4")

base_options = python.BaseOptions(model_asset_path=P_CHECKPT)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.FaceDetectorOptions.running_mode,
    output_segmentation_masks=True) 
detector = vision.PoseLandmarker.create_from_options(options)

# frame by frame predict and draw
annots = []
traj = []
for f in tqdm(video[200:350]):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(f))
    detection_result = detector.detect(image)
    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    traj.append(extract_landmark_trajectories(detection_result))
    # annots.append(annotated_image)

# save video 
# mediapy.write_video(f"results/1f2a37641863491588b98e1dfa4de062.mp4", annots, fps=30)

# make a pandas dataframe
import pandas as pd
df = pd.DataFrame(traj)
df.to_csv("results/1f2a37641863491588b98e1dfa4de062.csv", index=False)

mp_pose = mp.solutions.pose

# plot the 2D trajectories of the elbow, wrist and shoulder
import matplotlib.pyplot as plt

traj_np = np.array(traj)
t = np.arange(len(traj_np))

def lowpass_filter(data, alpha=0.90):
  filtered = [data[0]]
  for i in range(1, len(data)):
    filtered.append(alpha * data[i] + (1 - alpha) * filtered[-1])
  filtered = filtered[::-1]
  for i in range(1, len(data)):
    filtered[i] = alpha * filtered[i] + (1 - alpha) * filtered[i-1]
  filtered = filtered[::-1]   
  return filtered

# plotting
plt.clf()
plt.figure(figsize=(20, 10))
k = 0
for i in range(3):
  for j in range(4):
    plt.subplot(3, 4, k+1)
    plt.plot(t, lowpass_filter(traj_np[:, k, 0]), label="x")
    plt.plot(t, lowpass_filter(traj_np[:, k, 1]), label="y")
    plt.plot(t, lowpass_filter(traj_np[:, k, 2]), label="z")
    # plt title based on looping through the keys in lookup_Table
    # put centimeter on the y-axis
    plt.ylabel("d [m]")
    plt.xlabel("t [s]")
    # place the legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    # make more spacy between plots
    plt.tight_layout()
    plt.title(list(lookup_table.keys())[k])
    k += 1
  
# save fig
plt.savefig("results/1f2a37641863491588b98e1dfa4de062.png")