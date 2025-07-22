
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Accelerometer.csv')
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

roll = (df_gs['x'] / max_x) * 30  # max g equivalent angle  acceleration = 30°
pitch = (df_gs['y'] / max_y) * 30
x_trans = np.zeros_like(roll)
y_trans = np.zeros_like(roll)
z_gain = 500      # mm
yaw = np.zeros_like(roll)

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

pose_df.to_csv('pose_trajectory.csv', index=False)