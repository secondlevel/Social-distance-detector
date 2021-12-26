import cv2
import time
from matplotlib import colors
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from build.camera_calibrate_utils import rgb2gray_c, rgb2gray2_c, rgb2gray2_multithread_c
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


def plot_performance(image_number):
    
    image_dir = "image"

    BGR2GRAYTIME_MULTI = dict()

    image_path = "image/1.jpg"
    image = cv2.imread(image_path)
    
    print("Calculate the time that RGB to Gray ("+str(image_number)+" images).")
    print()
    t1 = time.time()
    
    for image_name in range(image_number):
        image_path = os.path.join(image_dir, str(image_name+1)+".jpg")
        # image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb2gray_p(image)
    
    t2 = time.time()
    print("python execution time: {} s".format(t2-t1))
    performance['python'].append(t2-t1)
    print()
    t1 = time.time()

    for image_name in range(image_number):
        image_path = os.path.join(image_dir, str(image_name+1)+".jpg")
        # image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb2gray_c(image)

    t2 = time.time()
    print("pybind c++ version 1 execution time: {} s".format(t2-t1))
    performance['pybind11_1'].append(t2-t1)
    print()
    t1 = time.time()

    for image_name in range(image_number):
        image_path = os.path.join(image_dir, str(image_name+1)+".jpg")
        # image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb2gray2_c(image)

    t2 = time.time()
    print("pybind c++ version 2 execution time: {} s".format(t2-t1))
    performance['pybind11_2'].append(t2-t1)
    print()
    t1 = time.time()

    for image_name in range(image_number):
        image_path = os.path.join(image_dir, str(image_name+1)+".jpg")
        # image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb2gray2_multithread_c(image)
    
    t2 = time.time()
    print("pybind c++ version 3 execution time: {} s".format(t2-t1))
    performance['pybind11_3'].append(t2-t1)
    print()
    t1 = time.time()

    for image_name in range(image_number):
        image_path = os.path.join(image_dir, str(image_name+1)+".jpg")
        # image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    t2 = time.time()
    print("opencv execution time: {} s".format(t2-t1))
    performance['opencv'].append(t2-t1)
    print()

if __name__ == "__main__":
    
    # performance = dict([["python",[]],["pybind11_1",[]],["pybind11_2",[]],["pybind11_3",[]],["opencv",[]]])

    # for number in range(1,21):
    #     plot_performance(number)
    
    # print(performance)

    x_tricks = [x for x in range(1,21)]
    # y_tricks = [0.001*y for y in range(1,11)]

    performance = dict({'python': [1.5749773979187012, 3.121262788772583, 4.702819585800171, 6.264744758605957, 7.785413503646851, 9.395434379577637, 11.119844198226929, 12.737167358398438, 14.19506573677063, 15.627485275268555, 17.346604108810425, 18.931814908981323, 20.4638934135437, 21.8944411277771, 23.497843980789185, 25.011369228363037, 26.669386625289917, 28.25062346458435, 30.415451288223267, 40.39329290390015], 'pybind11_1': [0.032533884048461914, 0.06439208984375, 0.0955817699432373, 0.12624526023864746, 0.15649938583374023, 0.19711780548095703, 0.22571182250976562, 0.25790858268737793, 0.2830190658569336, 0.3105607032775879, 0.34296703338623047, 0.37442731857299805, 0.4071354866027832, 0.4330306053161621, 0.46541666984558105, 0.5054941177368164, 0.5395078659057617, 0.5722496509552002, 0.6011626720428467, 0.6252665519714355], 'pybind11_2': [0.008468389511108398, 0.011549234390258789, 0.020992517471313477, 0.024053573608398438, 0.02918529510498047, 0.035472869873046875, 0.044551849365234375, 0.046608686447143555, 0.05716061592102051, 0.05694460868835449, 0.06218886375427246, 0.06816267967224121, 0.07518219947814941, 0.08140850067138672, 0.08457088470458984, 0.0886530876159668, 0.09702539443969727, 0.09924530982971191, 0.10755181312561035, 0.10985922813415527], 'pybind11_3': [0.008541107177734375, 0.017377614974975586, 0.0254056453704834, 0.03266119956970215, 0.041304588317871094, 0.04918718338012695, 0.05671811103820801, 0.06501269340515137, 0.07248687744140625, 0.0796670913696289, 0.0879054069519043, 0.09489893913269043, 0.10193276405334473, 0.11264634132385254, 0.1193842887878418, 0.12744545936584473, 0.13413548469543457, 0.14457917213439941, 0.15512919425964355, 0.1593327522277832], 'opencv': [0.003945112228393555, 0.008539915084838867, 0.010729074478149414, 0.013991355895996094, 0.01824665069580078, 0.022336721420288086, 0.028447389602661133, 0.02966165542602539, 0.03817629814147949, 0.041345834732055664, 0.04662179946899414, 0.04358363151550293, 0.05118727684020996, 0.05533576011657715, 0.05783724784851074, 0.060866355895996094, 0.06505465507507324, 0.06884121894836426, 0.07140278816223145, 0.07828974723815918]})
    
    plt.title("BGR2GRAY function performance")

    plt.plot(performance['python'], color='green', label='pure python version')
    plt.plot(performance['pybind11_1'], color='green', label='py::array_t version')
    plt.plot(performance['pybind11_2'], color='red', label='bufferinfo py::array_t version')
    plt.plot(performance['pybind11_3'], color='skyblue', label='bufferinfo py::array_t with multithread version')
    plt.plot(performance['opencv'], color='orange', label='opencv version')
    plt.legend(loc='best')
    plt.xticks(x_tricks)
    # plt.yticks(y_tricks)

    plt.xlabel("numebr of the image")
    plt.ylabel("seconds")
    plt.savefig("BGR2GRAY_PERFORMANCE.jpg")

    df = pd.DataFrame(performance).T
    df.columns = x_tricks
    df.index = ['python', 'array_t', 'bufferinfo', 'bufferinfo_multithread', 'opencv']
    df.to_csv("performance.csv", encoding="utf-8-sig")
    print(df)