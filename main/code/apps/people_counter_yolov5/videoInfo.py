import cv2

import yaml

with open('settings.yaml', 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

cap = cv2.VideoCapture(settings['video_source'])

if not cap.isOpened():
    print("Error opening video file")

else:
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f'Resolution: {width} x {height}')
    print(f'FPS: {fps}')
    print(f'Total Frames: {frame_count}')