import cv2 as cv
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt


def extract_frames(path, start_frame=0, stop_frame=0, invert=True, apply_func=None, step=1, *args, **kwargs):
    video = cv.VideoCapture(path)
    num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    if stop_frame == 0:
        stop_frame = num_frames
    frames = [None] * num_frames
    for i in range(start_frame, stop_frame):
        _, frame = video.read()
        # frame = to_8bit(frame)
        if invert: frame = skimage.util.invert(frame)
        if i % step == 0 and callable(apply_func):
            apply_func(path, frame, i, *args, **kwargs)
        frames[i] = frame / 256
    video.release()
    return frames

def play(frames, stop_key=" "):
    for frame in frames:
        cv.imshow('Example', frame)
        if cv.waitKey(20) & 0xFF==ord(stop_key):
            break
    cv.destroyAllWindows()
    
def save_frame(path_to_video, frame, i, target_dir=""):
    video_name = path_to_video.split("/")[-1]
    img_name = video_name.split(".")[0] + str(i) + ".png"
    img_path = target_dir + "/" + img_name
    cv.imwrite(img_path, frame)
    
def videos_to_images(video_folder):
    for video in os.listdir(video_folder):
        if video.endswith(".avi"):
            img_folder = video_folder + video.split(".")[0]
            if not os.path.exists(img_folder):    
                os.mkdir(img_folder)
            path_to_video = video_folder + video
            print(path_to_video)
            extract_frames(path_to_video, 0, 0, True, save_frame, 1, target_dir=img_folder)
            
def display(img):
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.imshow(img, cmap='gray')
    
def read(path):
    img = cv.imread(path)
    return img