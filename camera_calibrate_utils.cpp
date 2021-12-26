#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv_parameters.hpp"
#include <iostream>
#include <fstream>

#define NUM 2

using namespace std;
namespace py=pybind11;

int *ptr1_multithread=nullptr, *ptr2_multithread=nullptr;
int X_multithread,Y_multithread,Z_multithread;

void* calculate_multithread_grayvalue(void* args)
{
    // mydata *graydata = (mydata*) data; 
    vector<int>* lfpoint = static_cast<vector<int>*>(args);
    
    for (size_t idx = lfpoint->at(0); idx < lfpoint->at(1); idx++)
        for (size_t idy = lfpoint->at(2); idy < lfpoint->at(3); idy++)
            ptr2_multithread[Y_multithread*idx+idy] = (ptr1_multithread[Z_multithread*(Y_multithread*idx+idy)+0]*30+ptr1_multithread[Z_multithread*(Y_multithread*idx+idy)+1]*59+ptr1_multithread[Z_multithread*(Y_multithread*idx+idy)+2]*11+50)/100;

    pthread_exit(NULL); // exit child thread 
}

py::array_t<int> rgb2gray2_multithread_c(py::array_t<int> img_rgb)
{
    
    py::buffer_info buf1 = img_rgb.request();
    
    if(buf1.shape[2]!=3)
    {
      throw std::runtime_error("RGB image must have 3 channels!");
    }

    X_multithread = buf1.shape[0];
    Y_multithread = buf1.shape[1];
    Z_multithread = buf1.shape[2];

    int single_width = floor(Y_multithread/NUM);
    int single_height = floor(X_multithread/NUM);

    /*  allocate the buffer */
    py::array_t<int> result = py::array_t<int>(X_multithread*Y_multithread);

    py::buffer_info buf2 = result.request();

    ptr1_multithread = (int *) buf1.ptr;
    ptr2_multithread = (int *) buf2.ptr;
    
    // top height, down height, left width, right width 
    // vector<vector<int>> image_point;

    for(int i=0; i<=X_multithread; i+=single_height)
    {
        vector<int> tmp;
        for(int j=0; j<=Y_multithread; j+=single_width)
        {
            if(i+single_height<=(X_multithread))
            {
                tmp.push_back(i);
                tmp.push_back(i+single_height);
            }
            else if(i+single_height>(X_multithread) && i+single_height!=X_multithread)
            {
                tmp.push_back(i);
                tmp.push_back(X_multithread);
            }
            if(j+single_width<=(Y_multithread))
            {
                tmp.push_back(j);
                tmp.push_back(j+single_width);
            }
            else if(j+single_width>(Y_multithread) && j+single_width!=Y_multithread)
            {
                tmp.push_back(j);
                tmp.push_back(Y_multithread);
            }

            if(tmp[0]!=tmp[1] && tmp[2]!=tmp[3])
            {
                pthread_t t;
                pthread_create(&t, NULL, calculate_multithread_grayvalue, &tmp);
                pthread_join(t, NULL);
                // image_point.push_back(tmp);
                // block_num++;
            }
            // image_point.push_back(tmp);
            tmp.clear();
        }
    }

    result.resize({X_multithread,Y_multithread});

    return result;
}

py::array_t<double> rgb2gray_c(py::array_t<unsigned char>& img_rgb)
{
    if(img_rgb.ndim()!=3)
    {
      throw std::runtime_error("RGB image must have 3 channels!");
    }

    py::array_t<unsigned char> img_gray = py::array_t<unsigned char>(img_rgb.shape()[0]*img_rgb.shape()[1]);

    img_gray.resize({img_rgb.shape()[0],img_rgb.shape()[1]});

    auto rgb = img_rgb.unchecked<3>();
    auto gray = img_gray.mutable_unchecked<2>();

    for(int i=0; i<img_rgb.shape()[0]; i++)
    {
        for(int j=0; j<img_rgb.shape()[1]; j++)
        {
            auto R=rgb(i,j,0);
            auto G=rgb(i,j,1);
            auto B=rgb(i,j,2);
            auto GRAY=(R*30+G*59+B*11+50)/100;
            gray(i,j)=static_cast<unsigned char>(GRAY);
        } 
    }
    return img_gray;
}

py::array_t<int> rgb2gray2_c(py::array_t<int> img_rgb)
{

    py::buffer_info buf1 = img_rgb.request();

    if(buf1.shape[2]!=3)
    {
      throw std::runtime_error("RGB image must have 3 channels!");
    }

    int X = buf1.shape[0];
    int Y = buf1.shape[1];
    int Z = buf1.shape[2];

    /*  allocate the buffer */
    py::array_t<int> result = py::array_t<int>(X*Y);

    py::buffer_info buf2 = result.request();

    int *ptr1 = (int *) buf1.ptr,
        *ptr2 = (int *) buf2.ptr;

    for (size_t idx = 0; idx < X; idx++)
        for (size_t idy = 0; idy < Y; idy++)
            ptr2[Y*idx+idy] = (ptr1[Z*(Y*idx+idy)+0]*30+ptr1[Z*(Y*idx+idy)+1]*59+ptr1[Z*(Y*idx+idy)+2]*11+50)/100;
    
    // reshape array to match input shape
    result.resize({X,Y});

    return result;
}

cv::Mat GrayNumpy2Mat(py::array_t<unsigned char>& img)
{
    if(img.ndim()!=2)
        throw std::runtime_error("image must have 2 dimension.");
    
    py::buffer_info buf = img.request();
    // cv::Mat mat(static_cast<int>(buf.shape[0]), static_cast<int>(buf.shape[1]), CV_8UC1, (unsigned char*) buf.ptr);
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*) buf.ptr);
    
    return mat;
}

cv::Mat RGBNumpy2Mat(py::array_t<unsigned char>& img)
{
    if(img.ndim()!=3)
        throw std::runtime_error("image must have 3 dimension.");
    
    py::buffer_info buf = img.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*) buf.ptr);

    return mat;
}

py::array_t<unsigned char> GrayMat2Numpy(cv::Mat& img)
{
    py::array_t<unsigned char> dst = py::array_t<unsigned char>({img.rows, img.cols}, img.data);
    return dst;
}

py::array_t<unsigned char> RGBMat2Numpy(cv::Mat& img)
{
    py::array_t<unsigned char> dst = py::array_t<unsigned char>({img.rows, img.cols, 3}, img.data);
    return dst;
}

py::array_t<double> D1Mat2Numpy(cv::Mat& array)
{

    // std::cout << "rows:" << array.rows << std::endl;
    // std::cout << "cols:" << array.cols << std::endl;
    // std::cout << "channels:" << array.channels() << std::endl;
    // std::cout << (double*)array.data << std::endl;

    py::array_t<double> dst = py::array_t<double>({array.rows, array.cols*array.channels()}, (double*)array.data);

    return dst;
}

class CameraCalibrate
{
    public:

        CameraCalibrate()
        {
            chessboard_rows = 7;
            chessboard_cols = 7;
        };

        CameraCalibrate(int rows, int cols)
        {
            chessboard_rows = rows;
            chessboard_cols = cols;
        };

        ~CameraCalibrate()
        {
            objpoints.clear();
            imgpoints.clear();
            objp.clear();
            corner_pts.clear();
            images.clear();
        };
        
        void GenerateCalibrateMatrix(std::string directoryname)
        {
            // int number = 1;

            for(int i{0}; i<chessboard_rows; i++)
            {   
                for(int j{0}; j<chessboard_cols; j++)
                    objp.push_back(cv::Point3f(j,i,0));   
            }

            // Path of the folder containing checkerboard images
            std::string path = "./"+directoryname+"/*.jpg";

            cv::glob(path, images);

            // Looping over all the images in the directory
            for(int i{0}; i<images.size(); i++)
            {
                frame = cv::imread(images[i]);
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

                // Finding checker board corners
                // If desired number of corners are found in the image then success = true 
                success = cv::findChessboardCorners(gray, cv::Size(chessboard_cols, chessboard_rows), corner_pts, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

                if(success)
                {
                    
                    cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
                    
                    // refining pixel coordinates for given 2d points.       
                    cv::cornerSubPix(gray, corner_pts, cv::Size(11,11), cv::Size(-1,-1), criteria);
                    
                    // Displaying the detected corner points on the checker board        
                    // cv::drawChessboardCorners(frame, cv::Size(chessboard_cols, chessboard_rows), corner_pts, success);
                    
                    objpoints.push_back(objp);      
                    imgpoints.push_back(corner_pts);
                }
            }

            // cv::destroyAllWindows();
            cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

        };

        py::array_t<unsigned char> ImageUndistort(std::string imagename)
        {   
            std::fstream foo;
            foo.open(imagename);
            if(foo.is_open() == false)
            {
                throw std::runtime_error("The image file does not exist.");
            }
            else
            {
                originalimage = cv::imread(imagename);
                if(!cameraMatrix.data || !distCoeffs.data)
                {
                    throw std::runtime_error("Please check that you have execute the GenerateCalibrateMatrix function successfully.");
                }
                else
                {
                    cv::undistort(originalimage, undistortimage, cameraMatrix, distCoeffs);
                    return RGBMat2Numpy(undistortimage);
                }
            }
        }

        py::array_t<unsigned char> ImageUndistort(py::array_t<unsigned char>& imageMatrix)
        {
            if(imageMatrix.ndim()!=3)
            {
                throw std::runtime_error("Please ensure that image matrix has value.");
            }
            else
            {
                if(!cameraMatrix.data || !distCoeffs.data)
                {
                    throw std::runtime_error("Please check that you have execute the GenerateCalibrateMatrix function.");
                }
                else
                {
                    originalimage = RGBNumpy2Mat(imageMatrix);
                    cv::undistort(originalimage, undistortimage, cameraMatrix, distCoeffs);
                    return RGBMat2Numpy(undistortimage);
                }
            }
        }

        void SaveUndistortedImage(std::string saveimagename)
        {
            if(!undistortimage.data)
            {
                throw std::runtime_error("Please check that you have execute the ImageUndistort function successfully.");
            }
            else
            {
                cv::imwrite(saveimagename, undistortimage);
            }
        }

        void SaveImage(std::string saveimagename, py::array_t<unsigned char>& saveimage)
        {
            if(saveimage.ndim()!=3)
            {
                throw std::runtime_error("The image array does not exist.");
            }
            else
            {
                cv::imwrite(saveimagename, RGBNumpy2Mat(saveimage));
            }
        }

        void ShowCalibrateResult()
        {
            if(!originalimage.data || !undistortimage.data)
            {
                throw std::runtime_error("Please check that you have execute the ImageUndistort function successfully.");
            }
            else
            {
                cv::imshow("origin", originalimage);
                cv::imshow("calibrate",undistortimage);
                cv::waitKey(0);
            }
        }

        void ShowImage(std::string windowsname, py::array_t<unsigned char>& showimage)
        {
            if(showimage.ndim()!=3 || showimage.ndim()!=2)
            {
                throw std::runtime_error("Please check that you have execute the ImageUndistort function successfully.");
            }
            else
            {
                cv::imshow(windowsname, RGBNumpy2Mat(showimage));
                cv::waitKey(0);
            }
        }
         
        py::array_t<double> GetCameraMatrix()
        {
            if(!cameraMatrix.data)
            {
                throw std::runtime_error("Please check that you have execute the GenerateCalibrateMatrix function.");
            }
            else
            {
                return D1Mat2Numpy(cameraMatrix);
            }
        }

        py::array_t<double> GetDistCoeffs()
        {
            if(!distCoeffs.data)
            {
                throw std::runtime_error("Please check that you have execute the GenerateCalibrateMatrix function.");
            }
            else
            {
                return D1Mat2Numpy(distCoeffs);
            }
        }

        py::array_t<double> GetRotationVector()
        {
            if(!R.data)
            {
                throw std::runtime_error("Please check that you have execute the GenerateCalibrateMatrix function.");
            }
            else
            {
                return D1Mat2Numpy(R);
            }
        }

        py::array_t<double> GetTranslationVector()
        {
            if(!T.data)
            {
                throw std::runtime_error("Please check that you have execute the GenerateCalibrateMatrix function.");
            }
            else
            {
                return D1Mat2Numpy(T);
            }
        }

        py::array_t<unsigned char> GetImageNumpy(std::string imagename)
        {
            
            std::fstream foo;
            foo.open(imagename);
            if(foo.is_open() == false)
            {
                throw std::runtime_error("Please check that you have this image file.");
            }
            else
            {
                cv::Mat image = cv::imread(imagename);
                return RGBMat2Numpy(image);
            }
        }

        // py::array_t<unsigned char> GetUndistortImage()
        // {
        //     if(!undistortimage.data)
        //     {
        //         throw std::runtime_error("Please check that you have execute the ImageUndistort function.");
        //     }
        //     else
        //     {
        //         return RGBMat2Numpy(undistortimage);
        //     }
        // }

    private:

        bool success;

        int chessboard_rows,chessboard_cols;
        
        // Creating vector to store vectors of 3D points for each checkerboard image
        std::vector<std::vector<cv::Point3f>> objpoints;
        
        // Creating vector to store vectors of 2D points for each checkerboard image
        std::vector<std::vector<cv::Point2f>> imgpoints;
        
        // Defining the world coordinates for 3D points
        std::vector<cv::Point3f> objp;

        // Extracting path of individual image stored in a given directory
        std::vector<cv::String> images;

        // vector to store the pixel coordinates of detected checker board corners
        std::vector<cv::Point2f> corner_pts;

        cv::Mat frame, gray, cameraMatrix, distCoeffs, R, T, originalimage, undistortimage, empty;
        py::array_t<unsigned char> empty_array_unsigned_char;
        py::array_t<double> empty_array;
        
};

PYBIND11_MODULE(camera_calibrate_utils, m)
{
    m.doc() = "image process function";
    m.def("rgb2gray_c", &rgb2gray_c, "function to transform RGB to Gray(auto)");
    m.def("rgb2gray2_c", &rgb2gray2_c, "function to transform RGB to Gray(buffer info)");
    m.def("rgb2gray2_multithread_c", &rgb2gray2_multithread_c, "transform bgr to gray with multithread");

    // m.def("GrayNumpy2Mat", &GrayNumpy2Mat);
    // m.def("RGBNumpy2Mat", &RGBNumpy2Mat);
    // m.def("GrayMat2Numpy", &GrayMat2Numpy);
    // m.def("RGBMat2Numpy", &RGBMat2Numpy);

    py::class_<CameraCalibrate>(m, "CameraCalibrate")
        .def(py::init<int, int>())
        .def("GetImageNumpy", &CameraCalibrate::GetImageNumpy)
        .def("GenerateCalibrateMatrix", &CameraCalibrate::GenerateCalibrateMatrix)
        .def("ImageUndistort", py::overload_cast<std::string>(&CameraCalibrate::ImageUndistort))
        .def("ImageUndistort", py::overload_cast<py::array_t<unsigned char>&>(&CameraCalibrate::ImageUndistort))
        .def("SaveUndistortedImage", &CameraCalibrate::SaveUndistortedImage)
        .def("SaveImage", &CameraCalibrate::SaveImage)
        .def("ShowCalibrateResult", &CameraCalibrate::ShowCalibrateResult)
        .def("ShowImage", &CameraCalibrate::ShowImage)
        .def("GetCameraMatrix", &CameraCalibrate::GetCameraMatrix)
        .def("GetDistCoeffs", &CameraCalibrate::GetDistCoeffs)
        .def("GetRotationVector", &CameraCalibrate::GetRotationVector)
        .def("GetTranslationVector", &CameraCalibrate::GetTranslationVector);
        // .def("GetUndistortImage", &CameraCalibrate::GetUndistortImage);
}