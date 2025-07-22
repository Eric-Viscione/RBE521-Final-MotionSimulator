import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randint, random
import pandas as pd


def rotation_matrix(angles):
    roll, pitch, yaw = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def load_poses(file):
    df = pd.read_csv(file)
    print(df.columns)
    seconds_elapsed = df['seconds_elapsed'] ##loaded in seconds
    dt = int(float('%.4f'%df['seconds_elapsed'].diff().abs().min())*1000) ##Convert to ms
    print(f"dt of this system is {dt}")
    needed_poses = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    poses_df = df[needed_poses].copy()
    poses_df_smooth = poses_df.rolling(window=10, center=True, min_periods=1).mean()
    poses = poses_df_smooth.to_numpy()
    return poses, dt
def random_pose():
    poses = [[0, 0, 750, 0, 0, 0]]

    for i in range(25):
        prev = poses[-1]
        random_x = (randint(0, 50) - 15)
        x = prev[0] + random_x
        if x > l_max:
            x = prev[0] - random_x
        random_y = (randint(0, 50) - 15)
        y = prev[1] + random_y
        if y > l_max:
            y = prev[1] - random_x
        random_z = (randint(0, 50) - 15)
        z = prev[2] + random_z
        if z > l_max:
            z = prev[2] - random_z
        random_pitch = (random()*2 - 1)
        random_roll = (random()*2 - 1)
        random_yaw = (random()*2 - 1)

        pitch = prev[3] + random_pitch
        roll = prev[4] + random_roll
        yaw = prev[5] + random_yaw
        if abs(pitch) > .8:
            pitch = prev[3] - random_pitch
        if abs(roll) > .8:
            roll = prev[4] - random_roll
        if abs(pitch) > .8:
            roll = prev[5] - random_roll
        poses.append([x, y, z, pitch, roll, yaw])
    x_vals = [pose[0] for pose in poses]
    pitch_vals = [pose[4] for pose in poses]
    fig, ax = plt.subplots()
    ax.plot(pitch_vals)
    plt.show()


def compute_leg_lengths(base_pts, platform_pts_transformed):
    return np.linalg.norm(platform_pts_transformed - base_pts, axis=0)
