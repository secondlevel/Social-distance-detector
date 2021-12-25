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

    # def test_exist_calibrate_directory(self):
    #     self.calibrator.GenerateCalibrateMatrix("image")
    
    # def test_same_calibrate_matrix(self):

    #     h,w = self.img.shape[:2]
    #     mtx = self.calibrator.GetCameraMatrix()
    #     dist = self.calibrator.GetDistCoeffs()
    #     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    #     python_result = cv2.undistort(self.img, mtx, dist, None, newcameramtx)

    #     self.calibrator.ImageUndistort("image/1.jpg")
    #     c_result = self.calibrator.GetUndistortImage()
    #     # assert((False in (python_result == c_result))==False)