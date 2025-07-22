
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
exaggeration_factor = 2.0

def exaggerate(value, factor):
    return np.sign(value) * (np.abs(value) ** 1.2) * factor

def create_poses(file, output_file):
    df = pd.read_csv(file)
    df.index = pd.to_datetime(df['time'], unit='ns')

    print(df.columns)
    seconds_elapsed = df['seconds_elapsed'] ##loaded in seconds
    dt = float('%.4f'%df['seconds_elapsed'].diff().abs().min())*1000 ##Convert to ms
    print(f"dt of this system is {dt}")
    df_gs = df.copy()
    axis = ['x', 'y', 'z']
    for a in axis:
        df_gs[a] = df[a] / 9.8  # Convert to g's

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  
    time_seconds = (df_gs.index - df_gs.index[0]).total_seconds()
    for i, a in enumerate(axis):
        axs[i].plot(time_seconds, df_gs[a])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_title(f'{a.upper()} Acceleration')
        axs[i].set_ylabel('Acceleration (g)')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    max_x = df_gs['x'].abs().max()
    max_y = df_gs['y'].abs().max()

    # roll = (df_gs['x'] / max_x) * 30  # max g equivalent angle  acceleration = 30°
    # pitch = (df_gs['y'] / max_y) * 30
    roll = exaggerate((df_gs['x'] / max_x) * 30, exaggeration_factor)
    pitch = exaggerate((df_gs['y'] / max_y) * 30, exaggeration_factor)
    # x_trans = np.zeros_like(roll)
    # y_trans = np.zeros_like(roll)
    xy_jitter_gain = 100 
    raw_x = (np.random.rand(len(df_gs)) - 0.5) * 2 * xy_jitter_gain
    raw_y = (np.random.rand(len(df_gs)) - 0.5) * 2 * xy_jitter_gain
    raw_yaw = (np.random.rand(len(df_gs)) - 0.5) * 2 * xy_jitter_gain
    # Smooth with Gaussian filter (sigma controls smoothness)
    x_trans = gaussian_filter1d(raw_x, sigma=10)
    y_trans = gaussian_filter1d(raw_y, sigma=10)
    yaw = gaussian_filter1d(raw_yaw, sigma=10)
    z_gain = 750      # mm
    # yaw = np.zeros_like(roll)

    z_offset = -1 * (df_gs['z'] - 1.0) * z_gain
    # Assemble full pose trajectory [X, Y, Z, Roll, Pitch, Yaw]
    pose_trajectory = np.vstack((seconds_elapsed, x_trans, y_trans, z_offset, roll, pitch, yaw)).T


    columns=['seconds_elapsed', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', ]
    pose_df = pd.DataFrame(pose_trajectory,columns=columns)
    pose_df['z'] = pose_df['z'].clip(lower=10, upper=1000)
    plt.figure(figsize=(10,6))
    plt.plot(df_gs.index, pose_df['roll'], label='Roll (deg)')
    plt.plot(df_gs.index, pose_df['pitch'], label='Pitch (deg)')
    plt.plot(df_gs.index, pose_df['z'], label='Z offset (mm)')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Pose Value")
    plt.title("Pose Trajectory Over Time")
    plt.grid(True)
    plt.show()

    for a in columns:
        diffs = pose_df[a].diff().abs()  
        max_delta = diffs.max()
        min_delta = diffs.min()
        print(f"Max Difference per dt in column {a.upper()}: {max_delta:.4f}")
        print(f"Min Difference per dt in column {a.upper()}: {min_delta:.4f}")

    pose_df.to_csv(output_file, index=False)

def main():
    accelerometer_file = 'Accelerometer.csv'
    output_file = 'pose_trajectory.csv'
    create_poses(accelerometer_file, output_file)

if __name__ == "__main__":
    main()
