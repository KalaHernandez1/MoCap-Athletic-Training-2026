import cv2
import numpy as np

cam_cal = np.load("C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\wetransfer_mocap-schooltest-slave2-zip_2026-03-28_2219\\MoCap SchoolTest-Master\\MoCap SchoolTest-Master\\CAM1_calibration_Work_Horse.npz")

size = cam_cal['resolution']
w = 800
h = 600
mtx = cam_cal['camera_matrix']
dist = cam_cal['dist_coeffs']

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

distorted_img = cv2.imread("C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\wetransfer_mocap-schooltest-slave2-zip_2026-03-28_2219\\MoCap SchoolTest-Master\\MoCap SchoolTest-Master\\schooltest1_session_001_2026-03-28_16-38-21\\frame_00512.jpg")

print(distorted_img.shape[:2])

# 3. Apply the map to the image
undistorted_img = cv2.undistort(distorted_img, mtx, dist, None, newcameramtx)

x,y,dx,dy = roi
undistorted_centered_img = undistorted_img[y:y+dy, x:x+dx]
# Display the image
cv2.imshow('Distorted Image', distorted_img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.imshow('Undistorted Image, Centered', undistorted_centered_img)

# Wait for a key press to close the window (0 means wait indefinitely)
cv2.waitKey(0)
cv2.destroyAllWindows()