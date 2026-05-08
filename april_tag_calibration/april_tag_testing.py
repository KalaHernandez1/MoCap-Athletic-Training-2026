from pupil_apriltags import Detector
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# AprilTag detector initialization, global initialization used in this case
my_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Takes np darray, R is 3x3, t is 3x1, converts to combined rotation and translation
def convert_3x3_to_4x4 (R, t):
    return np.concatenate((np.concatenate((R, t), axis=1), 
                           np.array([[0, 0, 0, 1.0]])), axis=0)

# Method to extract pose, utilizes a global detector declaration
# Parameter is the path to the image that needs to be scanned
# Prints helpful infor if printResults=true, 3x3 R and 3x1 t output
def extract_camera_poses_3x3 (img, camera_parameters, printResults=False):

    # reads in the image as grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detects any tags in the image
    results = my_detector.detect(img_gray, 
                    estimate_tag_pose=True, camera_params=camera_parameters, 
                    tag_size=0.173)
    
    # Print statement for testing, set printResults to true if desired
    if (printResults):
        for result in results:
            print("------------------------")
            print(result.pose_R, "\n", result.pose_t, "\n", np.linalg.norm(result.pose_t))
            print(result.pose_R.T, "\n", -(result.pose_R.T@result.pose_t), "\n", np.linalg.norm(-(result.pose_R.T@result.pose_t)))
            print("XXXXXXXXXXXXXXXXXXXXXXXX")
            print(result.pose_t, "\n", -(result.pose_R@(-(result.pose_R.T@result.pose_t))))
            print("------------------------")

    # code appends useful pose estimation data in list form
    # if there are multiple tags in frame, ids collected
    # so the poses can be coordinated later.
    output_ids = []
    output_camR = []
    output_camt = []
    output_err = []
    for result in results:
        output_ids.append(result.tag_id)
        output_camR.append(result.pose_R.T)
        output_camt.append(-(result.pose_R.T@result.pose_t))
        output_err.append(result.pose_err)

    # Return data as a tuple
    return output_ids, output_camR, output_camt, output_err

# Method to extract pose, utilizes a global detector declaration
# Parameter is the path to the image that needs to be scanned
# Prints helpful infor if printResults=true, 4x4 Rt output
def extract_camera_poses_4x4 (img, camera_parameters, printResults=False):

    # reads in the image as grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detects any tags in the image
    results = my_detector.detect(img_gray, 
                    estimate_tag_pose=True, camera_params=camera_parameters, 
                    tag_size=0.173)
    
    # Print statement for testing, set printResults to true if desired
    if (printResults):
        for result in results:
            print("------------------------")
            print(convert_3x3_to_4x4(result.pose_R.T, -(result.pose_R.T@result.pose_t)))
            print(np.linalg.inv(convert_3x3_to_4x4(result.pose_R, result.pose_t)))
            print("XXXXXXXXXXXXXXXXXXXXXXXX")
            print(result.pose_t, "\n", -(result.pose_R@(-(result.pose_R.T@result.pose_t))))
            print()
            print("------------------------")

    # code appends useful pose estimation data in list form
    # if there are multiple tags in frame, ids collected
    # so the poses can be coordinated later.
    output_ids = []
    output_camT = []
    output_err = []
    for result in results:
        output_ids.append(result.tag_id)
        # In this version, there is a total translational matrix, which is 4x4
        output_camT.append(np.linalg.inv(convert_3x3_to_4x4(result.pose_R, result.pose_t)))
        output_err.append(result.pose_err)

    # Return data as a tuple
    return output_ids, output_camT, output_err

def undistort_img(img_path, mtx, dist, newcameramtx, roi):
    distorted_img = cv2.imread(img_path)

    undistorted_img = cv2.undistort(distorted_img, mtx, dist, None, newcameramtx)

    x,y,dx,dy = roi
    undistorted_centered_img = undistorted_img[y:y+dy, x:x+dx]

    #return undistorted_centered_img
    return undistorted_centered_img

def setup_img_distortions(img_path, calibration_path):
    cam_cal = np.load(calibration_path)

    img = cv2.imread(img_path)

    mtx = cam_cal['camera_matrix']
    dist = cam_cal['dist_coeffs']

    h, w = img.shape[:2]
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    return mtx, dist, newcameramtx, roi


# Master = Cam1
# Slave 1 = Cam2
# Slave 2 = Cam3

base_path = "C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\wetransfer_mocap-schooltest-slave2-zip_2026-03-28_2219\\"
camera_path = ["MoCap SchoolTest-Master\\MoCap SchoolTest-Master\\",
               "MoCap SchoolTest-Slave1\\MoCap SchoolTest-Slave1\\",
               "MoCap SchoolTest-Slave2\\MoCap SchoolTest-Slave2\\"]
session_path = ["schooltest3_session_003_2026-03-28_16-41-14\\frame_",
                "schooltest3_session_003_2026-03-28_16-41-13\\frame_",
                "schooltest3_session_003_2026-03-28_16-41-13\\frame_"]
frame_number = "00010"
extension = ".jpg"
calibration_file = ["CAM1_calibration_Work_Horse.npz",
                    "CAM2_calibration_Work_Horse.npz",
                    "CAM3_calibration_Work_Horse.npz"]
                    

# Put image file path Here

img_paths = ["C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\frame_00107.jpg",
             "C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\frame_00107_1.jpg",
             "C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\frame_00107_2.jpg"]

calibration_paths = ["C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\calibration_files\\calibration_master_800x600.npz",
                     "C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\calibration_files\\calibration_slave1_800x600.npz",
                     "C:\\Users\\qkoen\\Desktop\\Course Work\\Spring2026\\Capstone\\calibration_files\\calibration_slave2_800x600.npz"]

camTs = []

for i in range(3):
    imgpath = base_path + camera_path[i] + session_path[i] + frame_number + extension
    #imgpath = img_paths[i]

    calibrationpath = calibration_paths[i]

    mtx, dist, newcameamtx, roi = setup_img_distortions(imgpath, calibrationpath)

    img = undistort_img(imgpath, mtx, dist, newcameamtx, roi)

    k_matrix = [newcameamtx[0][0],newcameamtx[1][1],newcameamtx[0][2],newcameamtx[1][2]]

    out_ids, out_CamT, out_err = extract_camera_poses_4x4(img, k_matrix)

    print(out_CamT)

    camTs.append(out_CamT[0])

    if i > 0:
        camTs[i] = np.linalg.inv(camTs[0]) @ camTs[i]

camTs[0] = np.linalg.inv(camTs[0]) @ camTs[0]

print (camTs)

points = []

for i in range(len(camTs)):
    points.append([])
    points[i].append(camTs[i]@np.array([0.0,0.0,0.0,1.0]))
    points[i].append(camTs[i]@np.array([0.0,0.0,0.25,1.0]))
    points[i].append(camTs[i]@np.array([0.0,0.0,1.0,1.0]))

#unfinished

data_points = []


for i in range(len(points)):
    data_points.append([])
    for j in range(3):
        data_points[i].append([])
        data_points[i][j] = np.array([points[i][0][j], points[i][1][j], points[i][2][j]])


# Plot 1
fig1, ax1 = plt.subplots(1, 1)
ax1 = fig1.add_subplot(projection='3d')
fig1.set_size_inches(6, 5)
ax1.plot(data_points[0][0], data_points[0][1], data_points[0][2], marker='.', linestyle='-', c='b')
ax1.plot(data_points[1][0], data_points[1][1], data_points[1][2], marker='.', linestyle='-', c='g')
ax1.plot(data_points[2][0], data_points[2][1], data_points[2][2], marker='.', linestyle='-', c='r')
#ax1.set_xlim(-1, 1)
#ax1.set_ylim(-1, 1)
#ax1.set_zlim(-1, 1)
ax1.set_title("Graphed position Vectors")
ax1.set_xlabel("X-Position (m)")
ax1.set_ylabel("Y-Position (m)")
ax1.grid(visible=True, which='major')
ax1.grid(visible=True, which='minor', color='grey',
         linestyle='--', linewidth=0.2)

plt.show()




#print(out_CamT[0])
#print(np.linalg.inv(out_CamT[0]))

#img2 = cv2.imread(imgpath)

#cv2.imshow("Distorted Image", img2)

#cv2.imshow("Undistorted Image", img)



# Wait for a key press to close the window (0 means wait indefinitely)
#cv2.waitKey(0)
#cv2.destroyAllWindows()