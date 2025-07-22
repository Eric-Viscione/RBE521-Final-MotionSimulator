import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math_helpers import rotation_matrix, compute_leg_lengths, load_poses
from random import randint, random
import pandas as pd


class stewart_platform:

    def __init__(self, base, platform, l_min, l_max):
        self.base = base
        self.platform = platform
        self.l_min = l_min
        self.l_max = l_max
        self.base_pts = self.get_base_points(self.base)
        self.platform_pts = self.get_platform_points(self.platform)  ##Points in the platforms local frame (will be transformed based off of ee pose)
        pass
    def get_base_points(self, radius=100):
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        return np.vstack([radius * np.cos(angles),
                        radius * np.sin(angles),
                        np.zeros(6)])
    def get_platform_points(self, radius=80):
        angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
        return np.vstack([radius * np.cos(angles),
                        radius * np.sin(angles),
                        np.zeros(6)])




# base_pts = get_base_points()
class animate:
    def __init__(self, robot):
        self.robot = robot

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Preplot base
        self.ax.plot(*np.column_stack((self.robot.base_pts, self.robot.base_pts[:, 0])), 'k-')
        self.ax.scatter(*self.robot.base_pts, c='blue', s=50)
        self.leg_lines = [self.ax.plot([], [], [], 'g--')[0] for _ in range(6)]
        self.top_line, = self.ax.plot([], [], [], 'r-')
        self.top_pts = self.ax.scatter([], [], [], c='red', s=100)

        self.ax.set_xlim([-robot.base*1.5, robot.base*1.5])
        self.ax.set_ylim([-robot.base*1.5, robot.base*1.5])
        self.ax.set_zlim([0, robot.l_max])
        self.ax.set_box_aspect([1, 1, 0.5])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Animated Stewart Platform")
        self.ax_leg = self.fig.add_axes([0.75, 0.15, 0.2, 0.6])  # bar chart axes
        self.bars = self.ax_leg.bar(np.arange(6), [0]*6, color='green')
        self.ax_leg.set_ylim([self.robot.l_min, self.robot.l_max])
        self.ax_leg.set_title("Leg Lengths")
        self.timer_text = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)
    def setup(self):
        self.top_line.set_data([], [])
        self.top_line.set_3d_properties([])
        self.top_pts._offsets3d = ([], [], [])
        for leg in self.leg_lines:
            leg.set_data([], [])
            leg.set_3d_properties([])
        return self.leg_lines + [self.top_line, self.top_pts]

    def update(self, frame, poses, dt):
        elapsed_time = frame * dt / 1000
        self.timer_text.set_text(f"Time: {elapsed_time:.2f} s")
        pose = poses[frame]
        x, y, z = pose[:3]
        
        roll, pitch, yaw = pose[3:]
        translation = np.array([x, y, z])
        # euler_angles = [roll, pitch, yaw]
        euler_angles = np.radians([roll, pitch, yaw])
        platform_transformed = self.transform_platform(self.robot.platform_pts, translation, euler_angles)
        leg_lengths = compute_leg_lengths(self.robot.base_pts, platform_transformed)

        # top platform
        self.top_line.set_data(platform_transformed[0], platform_transformed[1]) ##createsd the 2d upper platform
        self.top_line.set_3d_properties(platform_transformed[2])                    ##makes that created shape in 3d
        self.top_pts._offsets3d = (platform_transformed[0], platform_transformed[1], platform_transformed[2])

        #update each leg
        for i in range(6):
            self.leg_lines[i].set_data([self.robot.base_pts[0, i], platform_transformed[0, i]],
                                [self.robot.base_pts[1, i], platform_transformed[1, i]])
            self.leg_lines[i].set_3d_properties([self.robot.base_pts[2, i], platform_transformed[2, i]])
        for i, bar in enumerate(self.bars):
            bar.set_height(leg_lengths[i])

        return self.leg_lines + [self.top_line, self.top_pts]
    
    def transform_platform(self, points, translation, euler_angles):
        
        R = rotation_matrix(euler_angles)
        return R @ points + translation[:, np.newaxis]
    def run(self, poses, dt):
        ani = FuncAnimation(self.fig, lambda frame: self.update(frame, poses, dt), frames=len(poses), init_func=lambda: self.setup(), blit=False, interval=dt/100)
        plt.show()


def main():
    base = 650
    platform = 350
    l_min = 0
    l_max = 1000
    file = "pose_trajectory.csv"
    poses, dt = load_poses(file)
    
    robot = stewart_platform(650, 350, 0, 1000)
    animator = animate(robot)
    animator.run(poses, dt)
if __name__ == "__main__":
    main()