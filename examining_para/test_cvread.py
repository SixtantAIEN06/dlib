import cv2
path='/home/ywt01/codes/dlib_test/examining_para/../testSet/photo/ford001.jpg'
img=cv2.imread(path)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
