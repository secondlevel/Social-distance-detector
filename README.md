# Camera Calibrator

 ===========================
## Basic Information

**Github repository**: https://github.com/secondlevel/Social-distance-detector

## Problem to Solve

In recent years, there are more and more deep learning methods to obtain the coordinates of objects in images or videos, such as fast-rcnn, mask-rcnn or yolo, etc. However, if these methods are used to detect the distance between bounding boxes, you will need to rely on the pixel distance captured in the picture. In actually, these pixel distances are not as perfect as expected, and they can not be directly calculated. The image captured by camera may be as shown in the figure below. The picture on the left is the case of "Normal", the picture in the middle is the case of "Positive Radial Distortation", and the picture on the right is the case of "Negative Radial Distortation".


<p float="center">
  
  <img src="md_image/camera_calibration_normal.png" width="250" title="Normal image" hspace="30" />
  
  <img src="md_image/camera_calibration_positive.png" width="280" title="Positive radial image" hspace="30" />
  
  <img src="md_image/camera_calibration_negative.png" width="230" title="Negative radial image" hspace="30" /> 
  
</p>


So what this project needs to do is a **camera calibration** that distorts the above-mentioned image back to normal image. 

## Methods Description

The purpose of this project aims to calibrate the camera with a frame sequence containing a chessboard, so the input will be the image and the output will be the camera matrix. 

In order to achieve this goal, the methods can be divided into five parts:

1. Image Reader: read lots of frames from a directory.

2. Color Converter: Converts the image from RGB to grayscale.

3. Chessboard Finder: The chessboard coordinates will be retrieved from the picture.  

4. Camera Matrix Generator: The camera matrix is generated by chessboard coordinate points and real coordinate points.  

5. Image Undistort: The image will be undistorted by camera matrix and distorted cofficient. 

## Prospective Users

This tool is for those who need to calculate the distance in the picture and undistorted the image.

## System Architecture

<img src="md_image/architecture.png" width="170" title="System Architecture" hspace="400" />

## API Description

### CameraCalibrate  

Check the size of the chessboard you want to use and declare the constructor.  

- calibrator = CameraCalibrate(7,7)

**GetImageNumpy()** - Read an image name and return the image numpy array.

- image_array = calibrator.GetImageNumpy("image/1.jpg")

**GenerateCalibrateMatrix()** - Read a directory name(stored some image that named xxx.jpg) and generate the rotation vector, translation vector, intrinsic matrix, and distorted coefficient,.etc. 

- calibrator.GenerateCalibrateMatrix("image")

**ImageUndistort()** - Read an image name or image array and return the undistorted image numpy array.

- undistorted_image_array1 = calibrator.ImageUndistort("image/1.jpg")  
- undistorted_image_array2 = calibrator.ImageUndistort(image_array)

**SaveUndistortedImage()** - Read the name of the image you want to save. It will save the image array to the system, such as our undistorted_image_array1 or undistorted_image_array2.

- calibrator.SaveUndistortedImage("result.jpg")

**SaveImage()** - Read the save name and the image numpy array. It will save this array to the system.

- calibrator.SaveImage("myresult.jpg", undistorted_image_array1)    

**ShowCalibrateResult()** - It will display both the last read image and last undistorted image, as shown in the following.

- calibrator.ShowCalibrateResult()  

<p float="center">
  
  <img src="md_image/origin.jpg" width="250" title="original image" hspace="30" />
  
  <img src="md_image/result.jpg" width="250" title="undistorted result" hspace="30" />
  
</p>

**ShowImage()** - Read the window name and image numpy array you want to show.

- calibrator.ShowImage("result_window", undistorted_image_array1) 

### additional function


**rgb2gray_c()** - Read the array of image and return the array of gray images. There using the py::array_t to implement the method.

- gray_image_array = rgb2gray_c(image_array)

**rgb2gray2_c()** - Read the array of image and return the array of gray images. There using the bufferinfo to get the value of py::array_t.

- gray_image_array = rgb2gray2_c(image_array)

**rgb2gray2_multithread_c()** - Read the array of image and return the array of gray images. There using the bufferinfo to get the value py::array_t, and through the multithread to implement the method.

- gray_image_array = rgb2gray2_multithread_c(image_array)

## Engineering Infrastructure

### Version Control

- Git

### Programing Language

- c++
- python
- make
- cmake

### Make-Pybind-Pytest

- The build system were convert the c++ function into python function through make and pybind.

- The python code were tested by pytest.

### Project Step

This project will be completed by executing the following steps:

- [X] Complete the camera calibration function.

- [X] Python binding.

- [X] Final testing.

## Schedule

| Week | Schedule |                                                                                                     
| ------------- | ------------- |
| Week 1  | 1. Environment Construct<br />2. Familiar with the camera calibration method<br />3. Familiar with the Video split method<br /> |
| Week 2  | 1. Familiar with the tools<br />2. Familiar with the camera calibration method<br />3. Familiar with the Video split method<br /> |
| Week 3  | Implement VideoSplitter | 
| Week 3  | Implement VideoSplitter |
| Week 4  | Implement CameraCalibratior |
| Week 5  | Implement CameraCalibratior  |
| Week 6  | Testing function   |
| Week 7  | Build Workflow|
| Week 8  | Project Presentation  |

## References
1. [Camera calibration] https://github.com/opencv/opencv/blob/master/samples/cpp/calibration.cpp
