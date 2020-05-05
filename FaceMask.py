import cv2
import numpy as np 
import os
import configparser
import sys
import matplotlib.pyplot as plt
import xmltodict
import time, datetime
import torch 
import torchvision.models as models
import torchvision.transforms as transforms
import torchvsion.datasets as datasets


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
def Createdir(dir_path): 
	if  not os.path.exists(dir_path):
		os.mkdir(dir_path)

#Create Labels for training saveed in Train Image Folder
def croplabel(data_dir,home_dir,label_name,status=False):
	if status==True:
		for l in range(len(label_name)):
			ext = label_name[l][1]
			Aname = label_name[l][0]
			result,size = getAnnot(data_dir,Aname)
			for r in range(len(result)):
				name = result[r][0]
				bndbox = result[r][1]
				image_path = data_dir + 'images' + '/' + Aname + ext
				image = cv2.imread(image_path)
				image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
				if name=='good':
					roi = image[bndbox[0][1]:bndbox[1][1],bndbox[0][0]:bndbox[1][0]]
					dir_0 = home_dir + 'train/' + '0/'
					Createdir(dir_0)
					cv2.imwrite(dir_0+Aname+'.jpg',roi)
					print("Created First Directory")
				elif name=='bad':
					roi = image[bndbox[0][1]:bndbox[1][1],bndbox[0][0]:bndbox[1][0]]
					dir_1 = home_dir + 'train/' + '1/'
					Createdir(dir_1)
					cv2.imwrite(dir_1+Aname+'.jpg',roi)
					print("Created 2nd Directory ")

#Load pretrainied resnet model
def loadModel():
	model = models.resnet50(pretrainied=True)
	for layer,param in model.named_parameters():
		if 'layer4' not in layer:
			param.requires_grad=False

		model.fc = torch.nn.Sequential(torch.nn.Linear(2048,512),
			torch.nn.ReLu(),
			torch.nn.Dropout(0.2),
			torch.nn.Linear(512,2),
			torch.nn.LogSoftmax(dim=1))

	train_transforms = transforms.Compose([transforms.Resize(224,224),
		transforms.ToTensor(),
		transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		])
	return model, train_transforms

def DatasetSplit(train_dir,train_transforms):
	dataset = datasets.ImageFolder(train_dir,train=train_transforms)
	dataset_size = len(dataset)
	train_size = int(len(dataset*0.6))
	val_size = int(len(dataset*0.2))
	test_size = dataset_size - train_size - val_size
	train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
	print("Length of Dataset :",dataset_size)
	print("Lenght of train dataset : ",train_size)
	print("Length of Val Dataset :",val_size)
	print("Length of Test Size : ", test_size)


#Main Function
def main():
	config = configparser.RawConfigParser()
	config.read('/home/ubuntu/Desktop/Julian_Folder/Projects/Face-Mask-Detector/FaceMask.ini')
	data_dir = config.get('Directory','Data_dir')
	home_dir = config.get('Directory','Home_dir')
	label_name = getImageName(data_dir)
	print("labelNames: ",len(label_name))
	ShowImage(data_dir,label_name)
	train_dir = home_dir + 'train/'
	Createdir(train_dir)
	croplabel(data_dir,home_dir,label_name)
	model,train_transforms = loadModel()
	DatasetSplit(train_dir,train_transforms)


if __name__ == '__main__':
	main()


