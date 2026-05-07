import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# MediaPipe Pose connections
CONNECTIONS = [
    (11,13), (13,15),   # left arm
    (12,14), (14,16),   # right arm
    (23,25), (25,27),   # left leg
    (24,26), (26,28),   # right leg
    (11,12),            # shoulders
    (23,24),            # hips
    (11,23), (12,24)    # torso sides
]

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def animate_3d(df):

    frames = df['frame'].unique()
    joints = df['joint'].unique()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):

        ax.clear()
        frame_data = df[df['frame'] == frame]

        points = {}
        for joint in frame_data['joint'].unique():
            joint_data = frame_data[frame_data['joint'] == joint]
            x = joint_data['x'].values[0]
            y = joint_data['y'].values[0]
            z = joint_data['z'].values[0]
            points[joint] = np.array([x, y, z])

        coords = np.array(list(points.values()))
        center = coords.mean(axis=0)
        coords -= center           # center skeleton
        coords[:,2] *= 3          # scale Z

        # plot joints
        for coord in coords:
            ax.scatter(coord[0], coord[1], coord[2], color='black', s=20)

        # plot connections
        for j1, j2 in CONNECTIONS:
            if j1 in points and j2 in points:
                idx1 = list(points.keys()).index(j1)
                idx2 = list(points.keys()).index(j2)
                line = np.vstack([coords[idx1], coords[idx2]])
                ax.plot(line[:,0], line[:,1], line[:,2], color='blue')

        # equal aspect
        max_range = np.ptp(coords, axis=0).max() / 2
        mid = coords.mean(axis=0)
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.invert_yaxis()

    ani = FuncAnimation(fig, update, frames=frames, interval=30)
    plt.show()
