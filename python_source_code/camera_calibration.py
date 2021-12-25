import cv2
import numpy as np
import pickle
import os

#use frames to calculate every parameters
# def calculate_calibrate_parameters(objpoints, imgpoints, img_size):
#     #reprojection_error = None
#     packed_tmp = ret_tmp, mtx_tmp, dist_tmp, rvecs_tmp, tvecs_tmp = cv2.calibrateCamera(objpoints, imgpoints, 
#                                                                                         img_size, None, None)
#     all_error_tmp=[]
#     for i in range(len(objpoints)):
#         imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_tmp[i], tvecs_tmp[i], mtx_tmp, dist_tmp)
#         error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#         all_error_tmp.append(error) # all_error: 由每張frame各自的error所組成的array

#     mean_error_tmp = sum(all_error_tmp)/len(all_error_tmp)

#     return imgpoints, objpoints, packed_tmp, ret_tmp, mtx_tmp, dist_tmp, rvecs_tmp, tvecs_tmp, all_error_tmp, mean_error_tmp

# def save_parameters(param):
#     file_name = "calibrate_param.pickle"
#     file = open('./'+file_name, 'wb')
#     pickle.dump(fixed_param, file)   #save parameters
#     file.close()

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
corner_x, corner_y = 7, 7

objp = np.zeros((corner_x*corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)#[0 0 0],[1 0 0],[2 0 0]........[6 6 0]

for image_index in range(1,21):
    image_path = os.path.join("image", str(image_index)+".jpg")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

    imgpoints.append(corners)
    objpoints.append(objp)

img_size = (image.shape[1], image.shape[0])

ret_tmp, mtx_tmp, dist_tmp, rvecs_tmp, tvecs_tmp = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# fixed_param = {'img_points':imgpoints, 'ret':ret_tmp, 'mtx':mtx_tmp, 'dist':dist_tmp, 'rvecs':rvecs_tmp, 'tvecs':tvecs_tmp, \
#         'error':all_error_tmp, 'mean_error':mean_error_tmp}

image = cv2.imread("image/2.jpg")
h, w = image.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_tmp, dist_tmp, (w,h), 1, (w,h))
dst = cv2.undistort(image, mtx_tmp, dist_tmp, None, newcameramtx)
