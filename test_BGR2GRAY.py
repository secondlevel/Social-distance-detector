import cv2
import time
import numpy as np
import os
from build.camera_calibrate_utils import rgb2gray_c, rgb2gray2_c
# from build.camera_calibrate_utils import rgb2gray_c

def rgb2gray_p(img_rgb):
    
    if img_rgb.shape[2]!=3:
        print("image channels is 3, correct!")
    
    h,w,c = img_rgb.shape
    gray = np.zeros(shape=(h,w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            R=img_rgb[i,j,0]
            G=img_rgb[i,j,1]
            B=img_rgb[i,j,2]
            GRAY=(R*30+G*59+B*11+50)/100
            gray[i,j]=np.uint8(GRAY)

    return gray

if __name__ == "__main__":
    
    image_dir = "image"

    BGR2GRAYTIME_MULTI = dict()
    BGR2GRAYTIME_SINGLE = dict()

    print()
    print("--------------------------------------------------------------------------------------------")
    print("Calculate the time that RGB to Gray (1 images).")
    print()

    image_path = "image/1.jpg"
    image = cv2.imread(image_path)

    t1 = time.time()
    image_p = rgb2gray_p(image)
    t2 = time.time()
    print("python execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_SINGLE['python']=t2-t1
    print()

    t1 = time.time()
    image_c = rgb2gray_c(image)
    t2 = time.time()
    print("pybind11 version 1 execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_SINGLE['pybind11_1']=t2-t1
    print()

    t1 = time.time()
    image_c = rgb2gray2_c(image)
    t2 = time.time()
    print("pybind11 version 2 execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_SINGLE['pybind11_2']=t2-t1
    print()

    t1 = time.time()
    image_o = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t2 = time.time()
    print("opencv execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_SINGLE['opencv']=t2-t1
    print()

    BGR2GRAYTIME_SINGLE = {k: v for k, v in sorted(BGR2GRAYTIME_SINGLE.items(), key=lambda item: item[1])}
    print("The time order between three method from fast to slow: ", end=" ")
    time_order = list(BGR2GRAYTIME_SINGLE)
    for order in range(len(time_order)-1):
        print(time_order[order], ">", end=" ")
    print(time_order[order+1])
    print()

    print("If opencv gray image is equal to pybind11 gray image:", False in (image_o == image_c), "\n")
    print("If pybind11 gray image have any negative pixel value?", np.any(image_c < 0), "\n")
    print("If pybind11 gray image have any pixel value that larger than 255?", np.any(image_c > 255), "\n")
    
    print("--------------------------------------------------------------------------------------------")
    print("Calculate the time that RGB to Gray (20 images).")
    print()
    t1 = time.time()
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb2gray_p(image)
    
    t2 = time.time()
    print("python execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_MULTI['python']=t2-t1
    print()
    t1 = time.time()

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb2gray_c(image)

    t2 = time.time()
    print("pybind c++ version 1 execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_MULTI['pybind11_1']=t2-t1
    print()
    t1 = time.time()

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb2gray2_c(image)

    t2 = time.time()
    print("pybind c++ version 2 execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_MULTI['pybind11_2']=t2-t1
    print()
    t1 = time.time()

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    t2 = time.time()
    print("opencv execution time: {} s".format(t2-t1))
    BGR2GRAYTIME_MULTI['opencv']=t2-t1
    print()

    print("The time order between four method from fast to slow: ", end=" ")
    BGR2GRAYTIME_MULTI = {k: v for k, v in sorted(BGR2GRAYTIME_MULTI.items(), key=lambda item: item[1])}

    time_order = list(BGR2GRAYTIME_MULTI)
    for order in range(len(time_order)-1):
        print(time_order[order], ">", end=" ")
    print(time_order[order+1])
    print()