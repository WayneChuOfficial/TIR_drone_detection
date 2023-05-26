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

def getiou(box1,box2):
    #box x1,y1,x2,y2
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou

def main():
    root = './dataset/UAV_dataset' 
    model_dir = './result/UAV_dataset_DNANet_26_05_2023_02_25_52_wDS/mIoU__DNANet_UAV_dataset_epoch_6090.pth.tar'#模型存放地址
    subdir_list = os.listdir(os.path.join(root,'train'))#len = 150
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = DNANet(num_classes=1,input_channels=3, block=Res_CBAM_block, num_blocks = [2, 2, 2, 2], nb_filter = [16, 32, 64, 128, 256], deep_supervision=True)
    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model.to(device)
    
    print('Begin to calculate acc on dataset')
    num = 0
    acc_list = []
    for subdir in subdir_list:
        with open(root+'/train/'+subdir+'/IR_label.json', "r", encoding="utf-8") as f:
            labels = json.load(f)
        filename_list = sorted(os.listdir(os.path.join(root,'train',subdir)))
        if 'ints' in filename_list[0][-4:]:#有.ipynb_checkpoints隐藏文件夹
            filename_list = filename_list[1:]
        bar = tqdm(total = len(filename_list))
        expression1 = 0
        expression2 = 0
        T = 0
        T_star = 0
        
        for i in range(len(filename_list)):
            filename = filename_list[i]
            if('jpg' in filename[-3:] or 'JPG' in filename[-3:]):
                origin_img = cv2.imread(root+'/train/'+subdir+'/'+filename,cv2.IMREAD_COLOR)[:,:,::-1].copy()
                
                pt = 1
                vt = labels['exist'][i]
                delta = vt
                iou = 0
                
                h,w,c = origin_img.shape
                img = preprocess(origin_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img)[-1]
                    output = torch.sigmoid(output)
                output = output.cpu().detach().numpy().reshape(256,256)
                output = np.uint8(output*255)
                _, output = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
                if output.max() > 0:
                    pt = 0#预测框不为空
                    bboxs = mask_find_bboxs(output)[-1]
                    x2,y2,width2,height2 = int(bboxs[0]/256*w), int(bboxs[1]/256*h), int(bboxs[2]/256*w),int(bboxs[3]/256*h)
                if labels['exist'][i] == 1:
                    expression2 += pt*delta
                    T_star += 1
                    if pt == 0:#预测框不为空且label存在时计算iou
                        x1,y1,width1,height1 = labels["gt_rect"][i]
                        iou = getiou([x1,y1,x1+width1,y1+height1],[x2,y2,x2+width2,y2+height2])
                T += 1
                expression1 += iou*delta + pt*(1-delta)
                if T_star > 0:
                    bar.set_description('video:%s acc = %.4f'%(subdir,expression1/T - 0.2 * (expression2/T_star)**0.3))
                num += 1
                bar.update(1)
        if T_star > 0:
            acc_list.append(expression1/T - 0.2 * (expression2/T_star)**0.3)
        else:
            acc_list.append(expression1/T)
        #acc_list.append(expression1/T - 0.2 * (expression2/T_star)**0.3)
        
    print('processed on %d images' %num)
    with open('acc_result.txt','w',encoding = 'utf-8') as f:
        f.write(str(acc_list))
    print(np.average(np.array(acc_list)))

if __name__ == "__main__":
    main()
