import numpy as np
import os
import pandas as pd
from deeptrack.models import LodeSTAR
import trackpy
import vidio
import matplotlib.pyplot as plt


class Tracker:
    def __init__(self, root_folder, detection_model, scale, fps):
        self.root_folder = root_folder
        self.detection_model = LodeSTAR(input_shape=(None, None, 3))
        self.detection_model.load_weights(detection_model)
        self.scale = scale
        self.fps = fps

    def detect_frame(self, frame, alpha=0.5, beta=0.5, cutoff=0.99):    
        detections = self.detection_model.predict_and_detect(frame[np.newaxis], alpha=alpha, beta=beta, cutoff=cutoff, mode="quantile")[0]        
        return detections
    
    def detect_video(self, path, alpha=0.5, beta=0.5, cutoff=0.99):
        frames = vidio.extract_frames(path)
        detections = [self.detect_frame(frame, alpha=alpha, beta=beta, cutoff=cutoff) for frame in frames]
        return detections
    
    def link_particles(self, detections, min_duration=100, min_displacement=100, search_range=25, memory=5):
        spots = []
        for i, frame in enumerate(detections):
            for spot in frame:
                spots.append([i, spot[1], spot[0]])
        spots_df = pd.DataFrame(spots, columns=["frame", "x", "y"])
        tracks_df = trackpy.link_df(spots_df, search_range=search_range, memory=memory)
        return self.compute_track_properties(tracks_df, min_duration=min_duration, min_displacement=min_displacement)
        
    def compute_track_properties(self, tracks_df, min_duration=100, min_displacement=100):
        tracks = []
        ntracks = tracks_df["particle"].nunique()
        for track_id in range(ntracks):
            track = tracks_df[tracks_df["particle"] == track_id]
            if len(track) < min_duration or abs(track["x"].iloc[-1] - track["x"].iloc[0]) < min_displacement:
                continue
            track = track[['frame', 'x', 'y']]
            time = np.ravel(track['frame'].to_numpy()) / self.fps
            x_um = np.ravel(track['x'].to_numpy()) * self.scale
            y_um = np.ravel(track['y'].to_numpy()) * self.scale
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
    
    def detect_all(self, alpha=0.5, beta=0.5, cutoff=0.99):
        detections_path = self.root_folder + "detections\\"
        os.mkdir(detections_path)
        files = [os.path.join(self.root_folder, file) for file in os.listdir(self.root_folder) if file.endswith(".avi")]
        for file in files:
            detected = self.detect_video(file, alpha=alpha, beta=beta, cutoff=cutoff)
            np.save(detections_path + file.split("\\")[-1][:-4] + ".npy", np.array(detected, dtype=object), allow_pickle=True)
            
    def track_all(self, min_duration=100, min_displacement=100, search_range=25, memory=5):
        all_tracks_path = self.root_folder + "tracks\\"
        detections_path = self.root_folder + "detections\\"
        os.mkdir(all_tracks_path)
        files = [os.path.join(detections_path, file) for file in os.listdir(detections_path)]
        for file in files:
            video_tracks_path = all_tracks_path + file.split("\\")[-1][:-4]
            os.mkdir(video_tracks_path)
            detections = np.load(file, allow_pickle=True)
            tracks = self.link_particles(detections, min_duration=min_duration, 
                                         min_displacement=min_displacement, 
                                         search_range=search_range, memory=memory)
            for i, track in enumerate(tracks):
                track.to_csv(video_tracks_path + "\\" + str(i) + ".csv")
                
    def random_frame(self):
        files = [file for file in os.listdir(self.root_folder) if file.endswith(".avi")]
        file = np.random.choice(files)
        frames = vidio.extract_frames(os.path.join(self.root_folder, file))
        return frames[np.random.randint(len(frames))]
    
    def freq_to_v(self):
        f_to_v = dict()
        all_tracks_path = self.root_folder + "\\tracks"
        for dir in os.listdir(all_tracks_path):
            f_to_v[int(dir[:2])] = []
        for dir in os.listdir(all_tracks_path):
            video_tracks_path = all_tracks_path + "\\" + dir
            freq = int(dir[:2])
            for file in os.listdir(video_tracks_path):
                data = pd.read_csv(video_tracks_path + "\\" + file)
                xs = data["x_um"].to_numpy()
                ys = data["y_um"].to_numpy()
                ts = data["time"].to_numpy()
                initial_x = xs[0]
                initial_y = ys[0]
                final_x = xs[-1]
                final_y = ys[-1]
                duration = ts[-1] - ts[0]
                f_to_v[freq].append(np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2) / duration)
        return f_to_v
    
    def compute_stats(self, f_to_v):
        freqs = np.array(list(f_to_v.keys()))
        actual_freqs = freqs * 0.9
        mean_vs = np.array([np.array(f_to_v[freq]).mean() for freq in freqs])
        std_vs = np.array([np.array(f_to_v[freq]).std() for freq in freqs])
        expected_vs = actual_freqs * 10
        percent_errors = abs(mean_vs - expected_vs) / expected_vs * 100
        
        result = pd.DataFrame(columns=["freq", "actual_freq", "mean_v", "std_v", "expected_v", "percent_error"])
        result["freq"] = freqs
        result["actual_freq"] = actual_freqs
        result["mean_v"] = mean_vs
        result["std_v"] = std_vs
        result["expected_v"] = expected_vs
        result["percent_error"] = percent_errors
        
        result.to_csv(self.root_folder + "\\v_vs_f.csv")
        
        return result
    
    def plot_v_vs_f(self, ax, v_vs_f):
        ax.errorbar(v_vs_f["actual_freq"], v_vs_f["mean_v"], yerr=v_vs_f["std_v"], fmt='.', label="LodeSTAR & Trackpy")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Mean velocity (um/s)")
        ax.set_title("V vs F")
        ax.legend()        
        
