import os
import cv2
import pytest
import timeit
import numpy as np
from build.camera_calibrate_utils import CameraCalibrate

def init_parameters():
    '''
    .def("GenerateCalibrateMatrix", &CameraCalibrate::GenerateCalibrateMatrix)
    .def("ImageUndistort", py::overload_cast<std::string>(&CameraCalibrate::ImageUndistort))
    .def("ImageUndistort", py::overload_cast<cv::Mat>(&CameraCalibrate::ImageUndistort))
    .def("SaveUndistortedImage", &CameraCalibrate::SaveUndistortedImage)
    .def("ShowCalibrateResult", &CameraCalibrate::ShowCalibrateResult)
    .def("GetCameraMatrix", &CameraCalibrate::GetCameraMatrix)
    .def("GetDistCoeffs", &CameraCalibrate::GetDistCoeffs)
    .def("GetRotationVector", &CameraCalibrate::GetRotationVector)
    .def("GetTranslationVector", &CameraCalibrate::GetTranslationVector)
    .def("GetImageNumpy", &CameraCalibrate::GetImageNumpy)
    .def("GetUndistortImage", &CameraCalibrate::GetUndistortImage)
    '''
    return None

class TestCameraCalibrate:
    img = cv2.imread("image/1.jpg")
    calibrator = CameraCalibrate(7,7)

    def test_undistort_image_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.GetUndistortImage()
        assert "Please check that you have execute the ImageUndistort function." in str(excinfo.value)

    def test_GetImageNumpy_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.GetImageNumpy("1.jpg")
        assert "Please check that you have this image file." in str(excinfo.value)
    
    def test_GetTranslationVector_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.GetTranslationVector()
        assert "Please check that you have execute the GenerateCalibrateMatrix function." in str(excinfo.value)
    
    def test_GetRotationVector_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.GetRotationVector()
        assert "Please check that you have execute the GenerateCalibrateMatrix function." in str(excinfo.value)
    
    def test_GetDistCoeffs_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.GetDistCoeffs()
        assert "Please check that you have execute the GenerateCalibrateMatrix function." in str(excinfo.value)
    
    def test_GetCameraMatrix_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.GetCameraMatrix()
        assert "Please check that you have execute the GenerateCalibrateMatrix function." in str(excinfo.value)
    
    def test_ShowCalibrateResult_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.ShowCalibrateResult()
        assert "Please check that you have execute the ImageUndistort function successfully." in str(excinfo.value)
    
    def test_SaveUndistortedImage_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.SaveUndistortedImage("result.jpg")
        assert "Please check that you have execute the ImageUndistort function successfully." in str(excinfo.value)
    
    def test_ImageUndistort_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.ImageUndistort("result.jpg")
        assert "The image file does not exist." in str(excinfo.value)
    
        with pytest.raises(RuntimeError) as excinfo:
            self.calibrator.ImageUndistort("image/1.jpg")
        assert "Please check that you have execute the GenerateCalibrateMatrix function successfully." in str(excinfo.value)

    def test_GetImageNumpy(self):
        compare_image = self.calibrator.GetImageNumpy("image/1.jpg")
        assert ((False in (self.img == compare_image))==False)
    
    def test_calibrate_matrix(self):
        self.calibrator.GenerateCalibrateMatrix("image")
        self.calibrator.ImageUndistort("image/1.jpg")
        c_result = self.calibrator.GetUndistortImage()

        h,w = self.img.shape[:2]
        mtx = self.calibrator.GetCameraMatrix()
        dist = self.calibrator.GetDistCoeffs()
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        python_result = cv2.undistort(self.img, mtx, dist, None, newcameramtx)

        result = (c_result == python_result)

        assert(np.sum(c_result<0)==0)
        assert(np.sum(c_result>255)==0)
        assert((np.sum(result==False)/(np.sum(result==False)+np.sum(result==True)))*100 > 50)
    
    def test_calibrate_performance(self):
        setup_c='''
from build.camera_calibrate_utils import CameraCalibrate
        '''

        c_camera_calibrate_code='''
calibrator = CameraCalibrate(7,7) 
calibrator.GenerateCalibrateMatrix('image')
calibrator.ImageUndistort('image/1.jpg')
'''
        
        setup_python='''
import cv2
import numpy as np
import os
        '''

        python_camera_calibrate_code='''
objpoints = [] 
imgpoints = [] 
corner_x, corner_y = 7, 7

objp = np.zeros((corner_x*corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

for image_index in range(1,21):
    image_path = os.path.join("image", str(image_index)+".jpg")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

    imgpoints.append(corners)
    objpoints.append(objp)

img_size = (image.shape[1], image.shape[0])

ret_tmp, mtx_tmp, dist_tmp, rvecs_tmp, tvecs_tmp = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

image = cv2.imread("image/2.jpg")
h, w = image.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_tmp, dist_tmp, (w,h), 1, (w,h))
dst = cv2.undistort(image, mtx_tmp, dist_tmp, None, newcameramtx)
'''

        c_cameracalibrate = timeit.Timer(stmt=c_camera_calibrate_code, setup=setup_c)
        min_c_cameracalibrate = min(c_cameracalibrate.repeat(repeat=10, number=1))

        python_cameracalibrate = timeit.Timer(stmt=python_camera_calibrate_code, setup=setup_python)
        min_python_cameracalibrate = min(python_cameracalibrate.repeat(repeat=10, number=1))

        with open("./performance/performance_cameracalibrate.txt", "w") as fin:
            fin.writelines(f"c++ cameracalibrate pybind11 to python utils takes {min_c_cameracalibrate} seconds\n")
            fin.writelines(f"python cameracalibrate takes {min_python_cameracalibrate} seconds\n")
            fin.writelines("c++ cameracalibrate speed-up over python cameracalibrate %g x\n" %(min_python_cameracalibrate/min_c_cameracalibrate))