import cv2
import os
import numpy as np
import sys

image_folder = sys.argv[1]
video_name = f'{sys.argv[1]}/video.avi'

images = np.array([img for img in os.listdir(image_folder) if img.endswith(".jpeg")])
numbers = np.argsort([int(img[:-5]) for img in images])
images = images[numbers]

frame = cv2.imread(images[0])
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()