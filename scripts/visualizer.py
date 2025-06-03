from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# initialization
data_root = "../data/nuscenes"
nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

# get one sample and its radar and camera data
sample = nusc.sample[0]
radar_token = sample['data']['RADAR_FRONT']
cam_token = sample['data']['CAM_FRONT']

radar_data = nusc.get('sample_data', radar_token)
cam_data = nusc.get('sample_data', cam_token)

# get radar point cloud
radar_pc = RadarPointCloud.from_file(os.path.join(data_root, radar_data['filename']))

# get camera image
img_path = os.path.join(data_root, cam_data['filename'])
image = cv2.imread(img_path)

# project radar points onto camera image
from nuscenes.utils.geometry_utils import view_points

# get transformation matrix
radar_cs = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
ego_pose_radar = nusc.get('ego_pose', radar_data['ego_pose_token'])
ego_pose_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])

# points to project (first 3 points for visualization)
radar_points = radar_pc.points[:3, :]
# transform to ego radar -> global -> ego cam -> cam
# step 1: radar -> ego
radar2ego = transform_matrix(radar_cs['translation'], Quaternion(radar_cs['rotation']))
# step 2: ego -> global
ego2global_radar = transform_matrix(ego_pose_radar['translation'], Quaternion(ego_pose_radar['rotation']))
# step 3: global -> ego_cam
global2ego_cam = np.linalg.inv(transform_matrix(ego_pose_cam['translation'], Quaternion(ego_pose_cam['rotation'])))
# step 4: ego_cam -> cam
ego2cam = np.linalg.inv(transform_matrix(cam_cs['translation'], Quaternion(cam_cs['rotation'])))

# transform
radar_points = np.vstack((radar_points, np.ones((1, radar_points.shape[1]))))
radar_points = ego2cam @ global2ego_cam @ ego2global_radar @ radar2ego @ radar_points

# project onto camera image
cam_intrinsic = np.array(cam_cs['camera_intrinsic'])
radar_img_points = view_points(radar_points[:3, :], cam_intrinsic, normalize=True)

# for visualization
for i in range(radar_img_points.shape[1]):
    x, y = int(radar_img_points[0, i]), int(radar_img_points[1, i])
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Radar Points on Camera Image")
plt.axis('off')
plt.show()