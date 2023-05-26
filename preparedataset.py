import os
import cv2
import numpy as np
import random
import json
from tqdm import tqdm
import random

random.seed(41)

root = './dataset/UAV_dataset'
trainval_img_list = []
train_img_list = []
val_img_list = []
test_img_list = []
subdir_list = os.listdir(os.path.join(root,'train'))

print('Begin to prepare train and val set')
bar = tqdm(total = len(subdir_list))

for subdir in subdir_list:
    num = 0
    with open(root+'/train/'+subdir+'/IR_label.json', "r", encoding="utf-8") as f:
        labels = json.load(f)
        filename_list = sorted(os.listdir(os.path.join(root,'train',subdir)))
        if 'ints' in filename_list[0][-4:]:#有.ipynb_checkpoints隐藏文件夹
            filename_list = filename_list[1:]
        for i in range(len(filename_list)):
            filename = filename_list[i]
            if('jpg' in filename[-3:] or 'JPG' in filename[-3:]):
                if random.uniform(0,1) < 0.8:
                    train_img_list.append((f'{subdir}/{filename}'))
                else:
                    val_img_list.append((f'{subdir}/{filename}'))
                trainval_img_list.append((f'{subdir}/{filename}'))
                
                img = cv2.imread(root+'/train/'+subdir+'/'+filename,cv2.IMREAD_GRAYSCALE)
                mask = np.zeros_like(img)
                if labels['exist'][i] == 1:
                    x,y,width,height = labels["gt_rect"][i]
                    region = img[y:y+height,x:x+width]
                    if len(region) != 0:#label中有0 0 0 0的存在
                        region_mean = np.mean(region)
                        region[region >= region_mean] = 255
                        region[region < region_mean] = 0
                        mask[y:y+height,x:x+width] = region
                        assert np.max(region) > 0
                if not os.path.exists(root+'/mask/train_val/'+subdir+'/'):
                    os.makedirs(root+'/mask/train_val/'+subdir+'/')
                cv2.imwrite(root+'/mask/train_val/'+subdir+'/'+filename, mask)
                
                num += 1
    assert num == len(filename_list) - 1
    bar.update(1)



print('Begin to prepare test set')
subdir_list = os.listdir(os.path.join(root,'test'))
bar = tqdm(total = len(subdir_list))
for subdir in subdir_list:
    filename_list = sorted(os.listdir(os.path.join(root,'test',subdir)))
    if 'ints' in filename_list[0][-4:]:#有.ipynb_checkpoints隐藏文件夹
        filename_list = filename_list[1:]
    for filename in filename_list:
        if('jpg' in filename[-3:] or 'JPG' in filename[-3:]):
            test_img_list.append((f'{subdir}/{filename}'))
    bar.update(1)

bar = tqdm(total = len(trainval_img_list))
with open(root+'/trainval.txt','w',encoding='utf-8') as f:
    for img_name in trainval_img_list:
        f.write(img_name)
        f.write('\n')
        bar.update(1)

        
bar = tqdm(total = len(train_img_list))
with open(root+'/train.txt','w',encoding='utf-8') as f:
    for img_name in train_img_list:
        f.write(img_name)
        f.write('\n')
        bar.update(1)        

bar = tqdm(total = len(val_img_list))
with open(root+'/val.txt','w',encoding='utf-8') as f:
    for img_name in val_img_list:
        f.write(img_name)
        f.write('\n')
        bar.update(1) 
        
bar = tqdm(total = len(test_img_list))
with open(root+'/test.txt','w',encoding='utf-8') as f:
    for img_name in test_img_list:
        f.write(img_name)
        f.write('\n')
        bar.update(1)
