import os
import cv2
import pytest
import timeit
import numpy as np
from build.camera_calibrate_utils import rgb2gray_c, rgb2gray2_c, rgb2gray2_multithread_c

def return_result_auto():
    
    image_rgb = cv2.imread("./image/1.jpg")
    image_gray = cv2.imread("./image/1.jpg", 0)
    result = rgb2gray_c(image_rgb)
    
    return image_rgb, image_gray, result

def return_result_bufferinfo():
    
    image_rgb = cv2.imread("./image/1.jpg")
    image_gray = cv2.imread("./image/1.jpg", 0)
    result = rgb2gray2_c(image_rgb)
    
    return image_rgb, image_gray, result

def return_result_bufferinfo_multithread():
    
    image_rgb = cv2.imread("./image/1.jpg")
    image_gray = cv2.imread("./image/1.jpg", 0)
    bufferinfo_multithread_result = rgb2gray2_multithread_c(image_rgb)
    
    return image_rgb, image_gray, bufferinfo_multithread_result

class TestRGB2GRAY:
    
    origin_auto, image_gray_auto, result_auto = return_result_auto()
    origin_bufferinfo, image_gray_bufferinfo, result_bufferinfo = return_result_bufferinfo()
    origin_bufferinfo_singlethread, image_gray_bufferinfo_singlethread, result_bufferinfo_multithread = return_result_bufferinfo_multithread()

    # test transform BGR image to GRAY image that using the auto to read type python numpy
    def test_image_size_auto(self):
        assert self.origin_auto.shape[0] == self.result_auto.shape[0]
        assert self.origin_auto.shape[1] == self.result_auto.shape[1]

    def test_channel_number_auto(self):
        assert len(self.result_auto.shape) == 2
    
    def test_pixel_value_auto(self):
        assert False in (self.image_gray_auto==self.result_auto)

    # test transform BGR image to GRAY image that using the bufferinfo to read type python numpy
    def test_image_size_bufferinfo(self):
        assert self.origin_bufferinfo.shape[0] == self.result_bufferinfo.shape[0]
        assert self.origin_bufferinfo.shape[1] == self.result_bufferinfo.shape[1]

    def test_channel_number_bufferinfo(self):
        assert len(self.result_bufferinfo.shape) == 2
    
    def test_pixel_value_bufferinfo(self):
        assert False in (self.image_gray_bufferinfo==self.result_bufferinfo)

    # test transform BGR image to GRAY image that using the bufferinfo to read type python numpy and using multithread to calculate grayvalue
    def test_image_size_bufferinfo_multithread(self):
        assert self.origin_bufferinfo_singlethread.shape[0] == self.result_bufferinfo_multithread.shape[0]
        assert self.origin_bufferinfo_singlethread.shape[1] == self.result_bufferinfo_multithread.shape[1]

    def test_channel_number_bufferinfo_multithread(self):
        assert len(self.result_bufferinfo_multithread.shape) == 2
    
    def test_pixel_value_bufferinfo_multithread(self):
        assert False in (self.image_gray_bufferinfo_singlethread==self.result_bufferinfo_multithread)
    
    def test_single_image_performance(self):
        
        setup='''
import cv2 
image_rgb = cv2.imread("./image/1.jpg")
from build.camera_calibrate_utils import rgb2gray_c, rgb2gray2_c, rgb2gray2_multithread_c
'''
        opencv = timeit.Timer("cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)", setup=setup)
        min_opencv_time = min(opencv.repeat(repeat=200, number=1))

        pybind11_1_time = timeit.Timer("rgb2gray_c(image_rgb)", setup=setup)
        min_pybind11_1_time = min(pybind11_1_time.repeat(repeat=200, number=1))

        pybind11_2_time = timeit.Timer("rgb2gray2_c(image_rgb)", setup=setup)
        min_pybind11_2_time = min(pybind11_2_time.repeat(repeat=200, number=1))

        pybind11_3_time = timeit.Timer("rgb2gray2_multithread_c(image_rgb)", setup=setup)
        min_pybind11_3_time = min(pybind11_3_time.repeat(repeat=200, number=1))

        with open("./performance/performance_single_image_rgb2bgr.txt", "w") as fin:
            fin.writelines(f"opencv bgr to gray takes {min_opencv_time} seconds\n")
            fin.writelines(f"(single-thread) pybind11 using type auto bgr to gray takes {min_pybind11_1_time} seconds\n")
            fin.writelines(f"(single-thread) pybind11 using type buffinfo bgr to gray takes {min_pybind11_2_time} seconds\n")
            fin.writelines(f"(multithread) pybind11 using type buffinfo bgr to gray takes {min_pybind11_3_time} seconds\n")
            fin.writelines("(single-thread) opencv speed-up over pybind11 bufferinfo %g x\n" %(min_pybind11_2_time/min_opencv_time))
            fin.writelines("(multithread) opencv speed-up over pybind11 bufferinfo %g x\n" %(min_pybind11_3_time/min_opencv_time))
            fin.writelines("pybind11 speed-up bufferinfo over pybind11 auto %g x\n" %(min_pybind11_2_time/min_pybind11_1_time))

    def test_multi_image_performance(self):

        multiimage_opencv_bgr2gray_code = '''
def multi_image_opencv(image_list):

    for image in image_list:
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return None
'''

        multiimage_pybind11_auto_bgr2gray_code = '''
def multi_image_opencv(image_list):

    for image in image_list:
        rgb2gray_c(image, cv2.COLOR_BGR2GRAY)

    return None
'''

        multiimage_pybind11_bufferinfo_bgr2gray_code = '''
def multi_image_opencv(image_list):

    for image in image_list:
        rgb2gray2_c(image, cv2.COLOR_BGR2GRAY)

    return None
'''

        multiimage_pybind11_bufferinfo_multithread_bgr2gray_code = '''
def multi_image_opencv(image_list):

    for image in image_list:
        rgb2gray2_multithread_c(image, cv2.COLOR_BGR2GRAY)

    return None
'''
        
        setup='''
import os
import cv2
image_list = [cv2.imread(image) for image in os.listdir("./image")]
from build.camera_calibrate_utils import rgb2gray_c, rgb2gray2_c, rgb2gray2_multithread_c
        '''
        opencv = timeit.Timer(stmt=multiimage_opencv_bgr2gray_code, setup=setup)
        min_opencv_time = min(opencv.repeat(repeat=200, number=1))

        pybind11_1_time = timeit.Timer(stmt=multiimage_pybind11_auto_bgr2gray_code, setup=setup)
        min_pybind11_1_time = min(pybind11_1_time.repeat(repeat=200, number=1))

        pybind11_2_time = timeit.Timer(stmt=multiimage_pybind11_bufferinfo_bgr2gray_code, setup=setup)
        min_pybind11_2_time = min(pybind11_2_time.repeat(repeat=200, number=1))

        pybind11_3_time = timeit.Timer(stmt=multiimage_pybind11_bufferinfo_multithread_bgr2gray_code, setup=setup)
        min_pybind11_3_time = min(pybind11_3_time.repeat(repeat=200, number=1))

        with open("./performance/performance_multi_image_rgb2bgr.txt", "w") as fin:
            fin.writelines(f"opencv bgr to gray takes {min_opencv_time} seconds\n")
            fin.writelines(f"(single-thread) pybind11 using type auto bgr to gray takes {min_pybind11_1_time} seconds\n")
            fin.writelines(f"(single-thread) pybind11 using type buffinfo bgr to gray takes {min_pybind11_2_time} seconds\n")
            fin.writelines(f"(multithread) pybind11 using type buffinfo bgr to gray takes {min_pybind11_3_time} seconds\n")
            fin.writelines("(single-thread) opencv speed-up over pybind11 bufferinfo %g x\n" %(min_opencv_time/min_pybind11_2_time))
            fin.writelines("(multithread) opencv speed-up over pybind11 bufferinfo %g x\n" %(min_opencv_time/min_pybind11_3_time))
            fin.writelines("pybind11 speed-up bufferinfo over pybind11 auto %g x\n" %(min_pybind11_2_time/min_pybind11_1_time))