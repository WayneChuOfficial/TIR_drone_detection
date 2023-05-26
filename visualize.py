import matplotlib.pylab as plt
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import os
import json
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from tqdm import tqdm
from model.model_DNANet import  Res_CBAM_block
from model.model_DNANet import  DNANet

input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

def preprocess(img):
    img  = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
    img = input_transform(img)
    return img

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1]

model_dir = './result/UAV_dataset_DNANet_26_05_2023_02_25_52_wDS/mIoU__DNANet_UAV_dataset_epoch_6090.pth.tar'#模型存放地址
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DNANet(num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks = [2, 2, 2, 2], nb_filter = [16, 32, 64, 128, 256], deep_supervision=True)
model.load_state_dict(torch.load(model_dir,map_location=device)['state_dict'])
model.to(device)
origin_img = cv2.imread('dataset/UAV_dataset/test/20190925_124612_1_4/000060.jpg',cv2.IMREAD_COLOR)[:,:,::-1].copy()
h,w,c = origin_img.shape
img = preprocess(origin_img).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(img)[-1]
    output = torch.sigmoid(output)   
output = output.cpu().detach().numpy().reshape(256,256)
#output = output > 0.5
output = np.uint8(output*255)
_, output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(origin_img)
if output.max() > 0:
    temp = 5
    bboxs = mask_find_bboxs(output)[-1]
    x,y,width,height = int(bboxs[0]/256*w), int(bboxs[1]/256*h), int(bboxs[2]/256*w),int(bboxs[3]/256*h)
    x,y = max(x-temp,0),max(y-temp,0)
    if x + width + temp < w:
        width += temp
    if y + height +temp <h:
        height += temp
    print(x,y,width,height)
    plt.plot([x,x+width,x+width,x,x],[y,y,y+height,y+height,y],c='r')
plt.show()
plt.savefig('vislize.png',bbox_inches = 'tight')