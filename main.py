import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

input_path  = "/home/hemantk/Documents/colorization/Data_zoo/LaMem/lamem/images/"
Q = 256

def batchReadImages(dir_path):
	abSpace = np.zeros((Q,Q));
	numOfImg = 0;
	for root,directories,filenames in os.walk(input_path):
		if numOfImg > 1e6:
			break;
		for filename in filenames:
			file_path = os.path.join(root,filename);
			_,ext = os.path.splitext(file_path);
			if not ext == '.txt':
				numOfImg = numOfImg + 1;
				# print(file_path)
				print(numOfImg)
				image = cv2.cvtColor(cv2.imread(file_path),cv2.COLOR_BGR2LAB);
				tempa = np.floor(image[:,:,1].flatten().astype(float)).astype(np.uint8);
				tempb = np.floor(image[:,:,2].flatten().astype(float)).astype(np.uint8);
				# print(tempb,tempa)
				abSpace[tempa,tempb] = abSpace[tempa,tempb] + 1;

	return abSpace,numOfImg

def mapSpace2Vec(wSpace):
	return wSpace.flatten();

def calcWeights(abSpace,numOfImg):
	wSpace = 1/abSpace;
	weights = mapSpace2Vec(wSpace);


def main():
	abSpace,numOfImg = batchReadImages(input_path);
	np.save('abspace',abSpace)
	pspace = (255.0*abSpace.astype(float)/np.max(abSpace)).astype(np.uint8);
	print(abSpace)
	plt.contourf(np.log(abSpace/numOfImg));
	plt.show()

if __name__ == '__main__':
	main();