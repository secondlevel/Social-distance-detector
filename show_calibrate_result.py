from build.camera_calibrate_utils import CameraCalibrate
import numpy as np
import cv2

def test_calibrate_matrix():

    calibrator = CameraCalibrate(7,7)
    img = calibrator.GetImageNumpy("image/1.jpg")
    calibrator.GenerateCalibrateMatrix("image")
    calibrator.ImageUndistort("image/1.jpg")
    c_result = calibrator.GetUndistortImage()

    h,w = img.shape[:2]
    mtx = calibrator.GetCameraMatrix()
    dist = calibrator.GetDistCoeffs()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    python_result = cv2.undistort(img, mtx, dist, None, newcameramtx)

    result = (c_result == python_result)

    # print(np.sum(result==True))
    # print(np.sum(result==False))
    # print(np.sum(result==True)/(np.sum(result==False)+np.sum(result==True)))

    cv2.imshow("python version", python_result)
    cv2.imshow("c++ version", c_result)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_calibrate_matrix()
 