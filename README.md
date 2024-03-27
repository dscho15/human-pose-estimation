
# Guide:

1a) conda create -n human-pose-estimation python=3.10

1b) conda activate human-pose-estimation 

2) pip install -r requirements.txt

3) wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Problems

If you have problems with Mesa then:

conda install -c conda-forge libstdcxx-ng
