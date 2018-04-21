import cv2
import numpy as np

img = np.ones((256,256,3),dtype=np.uint8);
A = np.reshape(np.repeat(np.arange(0,256,dtype=np.uint8),256,axis=0),(256,256));
B = A.T;
for i in range(256):
	L = i*np.ones((256,256),dtype=np.uint8);
	img[:,:,0] = L;
	img[:,:,1] = A;
	img[:,:,2] = B;
	# imgrgb = cv2.cvtColor(img,cv2.COLOR_LAB2RGB);
	# cv2.imwrite('labimages/'+str(i)+'.jpg',imgrgb)