#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv_parameters.hpp"
// #include <sys/stat.h>
#include <iostream>

// using namespace std;
// using namespace cv;

int CHECKERBOARD[2]{7,7};

int main(void)
{

    // int number = 1;
    
    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f>> objpoints;
    
    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f>> imgpoints;
    
    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;

    for(int i{0}; i<CHECKERBOARD[1]; i++)
    {   
        for(int j{0}; j<CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j,i,0));   
    }

    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;

    // Path of the folder containing checkerboard images
    std::string path = "./image/*.jpg";

    cv::glob(path, images);

    cv::Mat frame, gray;

    // vector to store the pixel coordinates of detected checker board corners
    std::vector<cv::Point2f> corner_pts;

    bool success;
    // Looping over all the images in the directory
    for(int i{0}; i<images.size(); i++)
    {
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true 
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

        if(success)
        {
            cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            
            // refining pixel coordinates for given 2d points.       
            cv::cornerSubPix(gray, corner_pts, cv::Size(11,11), cv::Size(-1,-1), criteria);
            
            // Displaying the detected corner points on the checker board        
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            
            objpoints.push_back(objp);      
            imgpoints.push_back(corner_pts);
        }

        // cv::imshow("Image",frame);
        // cv::waitKey(0);

        // cv::imwrite("./chessboard_corner_image/"+std::to_string(number)+".png", frame);
        // number++;

    }

    // cv::destroyAllWindows();
    cv::Mat cameraMatrix, distCoeffs, R, T, undistortimage;
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

    cv::Mat originalimage = cv::imread("image/2.jpg");
    cv::undistort(originalimage, undistortimage, cameraMatrix, distCoeffs);

    // cv::imshow("Image",undistortimage);
    // cv::waitKey(0);
    
    return 0;
}