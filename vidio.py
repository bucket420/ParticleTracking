import cv2 as cv
import skimage
import matplotlib.pyplot as plt

def extract_frames(path, invert=True):
    video = cv.VideoCapture(path)
    num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    frames = [None] * num_frames
    for i in range(num_frames):
        _, frame = video.read()
        if invert: 
            frame = skimage.util.invert(frame)
        frames[i] = frame / 256
    video.release()
    return frames

def display(img):
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.imshow(img, cmap='gray')
    
def draw_tracks(tracks, img, color=(0, 255, 0), thickness=2):
    for track in tracks:
        for i in range(len(track) - 1):
            cv.line(img, (int(track['x'].iloc[i]), int(track['y'].iloc[i])), 
                    (int(track['x'].iloc[i+1]), int(track['y'].iloc[i+1])), color, thickness)
    return img