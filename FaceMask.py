import cv2
import numpy as np 
import os
import configparser
import sys
import matplotlib.pyplot as plt
import xmltodict
import time, datetime


#To get thre image names from the Label Name
def getImageName(data_dir):
	label_dir = str(data_dir) + 'labels'
	label_name = []
	image_dir = str(data_dir) + 'images'
	ext = ['.png','.jpeg','.jpg']
	for labels in os.listdir(label_dir):
		for img in os.listdir(image_dir):
				for e in range(len(ext)):
					if img.endswith(ext[e]):
						labelname = labels.split('.xml')[0]
						imgname = img.split(ext[e])[0]
						if labelname==imgname:
							label_name.append((imgname,ext[e]))
	return label_name

#To View the Image along with their Label
def ShowImage(data_dir,label_name,numV=4,status=False):
	if status==True:
		image_dir = str(data_dir) + 'images'
		for l in range(len(label_name)):
			ext = label_name[l][1]
			Aname = label_name[l][0]
			result,size = getAnnot(data_dir,Aname)
			print("Result : ", len(result))
			print("Size : ",len(size))
			image_path = image_dir + '/' + Aname + ext
			image=cv2.imread(image_path)
			image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			thick = int(sum(size)/400)
			for objs in result:
				name,bbox = objs
				epochtime = str(time.time())
				if name == 'good':
					cv2.rectangle(image,bbox[0],bbox[1],(0,255,0),thick)
					cv2.imwrite('/home/ubuntu/Desktop/Julian_Folder/Projects/Face-Mask-Detector/images/'+ Aname+'.jpg',image)
				elif name == 'bad':
					cv2.rectangle(image,bbox[0],bbox[1],(255,0,0),thick)
					cv2.imwrite('/home/ubuntu/Desktop/Julian_Folder/Projects/Face-Mask-Detector/images/'+ Aname+'.jpg',image)
				else:
					cv2.rectangle(image,bbox[0],bbox[1],(0,0,255),thick)
					cv2.imwrite('/home/ubuntu/Desktop/Julian_Folder/Projects/Face-Mask-Detector/images/'+ Aname+'.jpg',image)


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
		bndbox = [(int(item['bndbox']['xmin']),int(item['bndbox']['ymin'])),(int(item['bndbox']['xmax']),int(item['bndbox']['ymax']))]
		result.append((name,bndbox))
	size = [int(annot['annotation']['size']['width']),int(annot['annotation']['size']['height'])]
	return result,size


#Create Directory 
def Createdir(home_dir,dir_name):
	dir_path = home_dir + dir_name + '/'
	if  not os.path.exists(dir_path):
		os.mkdir(dir_path)

#Create Labels for training 
def croplabel(data_dir,label_name,status=True):
	if status==True:
		



def main():
	config = configparser.RawConfigParser()
	config.read('/home/ubuntu/Desktop/Julian_Folder/Projects/Face-Mask-Detector/FaceMask.ini')
	data_dir = config.get('Directory','Data_dir')
	home_dir = config.get('Directory','Home_dir')
	label_name = getImageName(data_dir)
	print("labelNames: ",len(label_name))
	ShowImage(data_dir,label_name)


if __name__ == '__main__':
	main()


