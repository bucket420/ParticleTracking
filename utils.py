import cv2 as cv
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt
import pandas as pd
from laptrack import LapTrack
import json
import trackpy

def extract_frames(path, invert=True, *args, **kwargs):
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
    
def detect_frame(frame, model, alpha=0.5, beta=0.5, cutoff=0.99):    
    detections = model.predict_and_detect(frame[np.newaxis], alpha=alpha, beta=beta, cutoff=cutoff, mode="quantile")[0]        
    return detections

def detect_video(path, model, alpha=0.5, beta=0.5, cutoff=0.99):
    frames = extract_frames(path)
    detections = [detect_frame(frame, model, alpha=alpha, beta=beta, cutoff=cutoff) for frame in frames]
    return detections

def link_particles(detected, scale, fps, min_duration=100, min_displacement=100, search_range=25, memory=5):
    spots = []
    for i, frame in enumerate(detected):
        for spot in frame:
            spots.append([i, spot[1], spot[0]])
    spots_df = pd.DataFrame(spots, columns=["frame", "x", "y"])
    
    tracks = []
    tracks_df = trackpy.link_df(spots_df, search_range=search_range, memory=memory)
    ntracks = tracks_df["particle"].nunique()
    for track_id in range(ntracks):
        track = tracks_df[tracks_df["particle"] == track_id]
        if len(track) < min_duration or abs(track["x"].iloc[-1] - track["x"].iloc[0]) < min_displacement:
            continue
        track = track[['frame', 'x', 'y']]
        time = np.ravel(track['frame'].to_numpy()) / fps
        x_um = np.ravel(track['x'].to_numpy()) * scale
        y_um = np.ravel(track['y'].to_numpy()) * scale
        
        delta_x = [None]
        delta_y = [None]
        vx = [None]
        vy = [None]
        v = [None]
        for i in range(len(track)):
            if i > 0:
                delta_x.append(x_um[i] - x_um[i-1])
                delta_y.append(y_um[i] - y_um[i-1])
                vx.append(delta_x[i] / time[i])
                vy.append(delta_y[i] / time[i])
                v.append(np.sqrt(vx[i]**2 + vy[i]**2))
        track['time'] = time
        track['x_um'] = x_um
        track['y_um'] = y_um
        track['delta_x'] = delta_x
        track['delta_y'] = delta_y
        track['vx'] = vx
        track['vy'] = vy
        track['v'] = v
        tracks.append(track)
    
    return tracks

def detect_all(path, model, alpha=0.5, beta=0.5, cutoff=0.99):
    detections_path = path + "\\detections"
    os.mkdir(detections_path)
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    for file in files:
        detected = detect_video(file, model, alpha=alpha, beta=beta, cutoff=cutoff)
        np.save(detections_path + "\\" + file.split("\\")[-1][:-4] + ".npy", np.array(detected, dtype=object), allow_pickle=True)   
             
def track_all(path, scale, fps, min_duration=100, min_displacement=100,
              search_range=25, memory=5):
    tracks_path = path + "\\tracks"
    detections_path = path + "\\detections"
    os.mkdir(tracks_path)
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files if file.endswith(".avi")]
    for file in files:
        video_tracks = tracks_path + "\\" + file.split("\\")[-1][:-4]
        os.mkdir(video_tracks)
        detected = np.load(detections_path + "\\" + file.split("\\")[-1][:-4] + ".npy", allow_pickle=True)
        tracks = link_particles(detected, scale=scale, fps=fps, 
                                min_duration=min_duration, min_displacement=min_displacement,
                                search_range=search_range, memory=memory)
        for i, track in enumerate(tracks):
            track.to_csv(video_tracks + "\\" + str(i) + ".csv")
         
def draw_tracks(tracks, img, color=(0, 255, 0), thickness=2):
    for track in tracks:
        for i in range(len(track) - 1):
            cv.line(img, (int(track['x'].iloc[i]), int(track['y'].iloc[i])), 
                    (int(track['x'].iloc[i+1]), int(track['y'].iloc[i+1])), color, thickness)
    return img
            
    