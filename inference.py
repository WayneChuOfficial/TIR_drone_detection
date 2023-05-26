import torch
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

def main():
    root = './dataset/UAV_dataset' 
    model_dir = './result/UAV_dataset_DNANet_26_05_2023_02_25_52_wDS/mIoU__DNANet_UAV_dataset_epoch_6090.pth.tar'#模型存放地址
    subdir_list = os.listdir(os.path.join(root,'test'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("./test_prediction/"):
        os.makedirs("./test_prediction/")
    
    model = DNANet(num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks = [2, 2, 2, 2], nb_filter = [16, 32, 64, 128, 256], deep_supervision=True)
    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model.to(device)
    
    print('Begin to infer on test dataset')
    bar = tqdm(total = len(subdir_list))
    num = 0
    for subdir in subdir_list:
        with open(root+'/test/'+subdir+'/IR_label.json', "r", encoding="utf-8") as f:
            labels = json.load(f)
        filename_list = sorted(os.listdir(os.path.join(root,'test',subdir)))
        if 'ints' in filename_list[0][-4:]:#有.ipynb_checkpoints隐藏文件夹
            filename_list = filename_list[1:]

        for i in range(1,len(filename_list)):#第一帧已经给出
            filename = filename_list[i]
            if('jpg' in filename[-3:] or 'JPG' in filename[-3:]):
                origin_img = cv2.imread(root+'/test/'+subdir+'/'+filename,cv2.IMREAD_COLOR)[:,:,::-1].copy()
                h,w,c = origin_img.shape
                img = preprocess(origin_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img)[-1]
                    output = torch.sigmoid(output)
                output = output.cpu().detach().numpy().reshape(256,256)
                output = np.uint8(output*255)
                _, output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
                if output.max() > 0:
                    bboxs = mask_find_bboxs(output)[-1]
                    x,y,width,height = int(bboxs[0]/256*w), int(bboxs[1]/256*h), int(bboxs[2]/256*w),int(bboxs[3]/256*h)
                    labels["res"].append([x,y,width,height])
                else:
                    labels["res"].append([])
                num += 1
        result = json.dumps(labels)        
        with open("./test_prediction/" + subdir + ".txt", "w", encoding="utf-8") as f:
            f.write(result)
        bar.update(1)
    print('infer on %d images' %num)

if __name__ == "__main__":
    main()