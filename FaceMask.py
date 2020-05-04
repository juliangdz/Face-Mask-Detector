import cv2
import numpy as np 
import os
import configparser
import sys
import matplotlib.pyplot as plt
import xmltodict


#To get thre image names from the Label Name
def getImageName(data_dir):
	label_dir = str(data_dir) + 'labels'
	label_name = []
	image_dir = str(data_dir) + 'images'
	for labels in os.listdir(label_dir):
		if labels.endswith('.xml'):
			lname = labels.split('.xml')[0]
			label_name.append(lname)
	return label_name

#To View the Image along with their Label
def ShowImage(data_dir,label_name,numV=4):
	image_dir = str(data_dir) + 'images'
	for img in os.listdir(image_dir):
		ext = str(img).split('.')[1]
		imgN = str(img).split('.')[0]
		print("The Extension for {} is {}".format(imgN,ext))
		if imgN in label_name:
			for i in range(numV):
				result,size = getAnnot(data_dir,label_name[i])
				print("Result : ", len(result))
				print("Size : ",len(size))
				image_path = image_dir + '/' + label_name[i] + '*'
				image=cv2.imread(image_path)
				image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
				thick = int(sum(size)/400)
				for objs in result:
					name,bbox = objs
					print(type(bbox[0]))
					print(bbox[1])
					print(type(thick))
					if name == 'good':
						cv2.rectangle(image,(bbox[0],bbox[1]),((bbox[0]+size[0]),(bbox[1]+size[1])),(0,255,0),thick)
					elif name == 'bad':
						cv2.rectangle(image,(bbox[0],bbox[1]),((bbox[0]+size[0]),(bbox[1]+size[1])),(255,0,0),thick)
					else:
						cv2.rectangle(image,(bbox[0],bbox[1]),((bbox[0]+size[0]),(bbox[1]+size[1])),(0,0,255),thick)

				plt.figure(figsize=(20,20))
				plt.subplot(1,2,1)
				plt.axis('off')
				plt.title(label_name[i])
				plt.imshow(image)


#Get the Boondung Box Coordinates of the Images 
def getAnnot(data_dir,imgN):
	label_path = data_dir + 'labels' + '/' + imgN + '.xml'
	annot = xmltodict.parse(open(label_path,'rb'))
	itemList = annot['annotation']['object']
	if not isinstance(itemList,list):
		itemList = [itemList]
	result=[]
	for item in itemList:
		name = item['name']
		bndbox = [(int(item['bndbox']['xmin'])),(int(item['bndbox']['ymin'])),(int(item['bndbox']['xmax'])),(int(item['bndbox']['ymax']))]
		result.append((name,bndbox))
	size = [int(annot['annotation']['size']['width']),int(annot['annotation']['size']['height'])]
	return result,size


def main():
	config = configparser.RawConfigParser()
	config.read('/home/ubuntu/Desktop/Julian_Folder/Mask/Face Mask /FaceMask.ini')
	data_dir = config.get('Directory','Data_dir')
	label_name = getImageName(data_dir)
	print("labelNames: ",len(label_name))
	ShowImage(data_dir,label_name)


if __name__ == '__main__':
	main()


