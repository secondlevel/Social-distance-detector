import pickle
import cv2

#type in file name (parameters used)
file_name1 = 't'
file = open('./{}/param.pickle'.format(file_name1),'rb')
fixed_param = pickle.load(file)
file.close()

mtx = fixed_param['mtx']
dist = fixed_param['dist']
#reprojection_error = fixed_param['reprojection_error']
print("file name: ", file_name1)
print(fixed_param)


""" 
#capture photo
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if ret:
            #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = frame
            cv2.imwrite('shoot({}).jpg'.format(file_name1), img)
            w = frame.shape[1]
            h = frame.shape[0]
            img_size = (w, h)
            cap.release()  
            cv2.destroyAllWindows()
            break
        else:
            break
"""
#upload picture
img = cv2.imread('shoot.jpg') #type in picture name
h,  w = img.shape[:2]

#undistort img
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imwrite('calib_result({}).jpg'.format(file_name1), dst)

