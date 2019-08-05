'''     
          ┌─┐       ┌─┐
       ┌──┘ ┴───────┘ ┴──┐
       │                 │
       │       ───       │
       │  ─┬┘       └┬─  │
       │                 │
       │       ─┴─       │
       │                 │
       └───┐         ┌───┘
           │         │
           │         │
           │         │
           │         └──────────────┐
           │                        │
           │  MADE IN YANGWENYING   ├─┐
           │                        ┌─┘    
           │                        │
           └─┐  ┐  ┌───────┬──┐  ┌──┘         
             │ ─┤ ─┤       │ ─┤ ─┤         
             └──┴──┘       └──┴──┘ 
                 神兽保佑 
                代码无BUG! 
'''
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
import torch.nn as nn
import os
from network import U_Net
from utils import search_file
import cv2
import numpy as np
from utils import  COLOR_DICT
from utils import dense_crf
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
num_channels = 3
batch_size = 4
size = (512, 384)
root = "membrane/test"
img_file = search_file(root, [".jpg"])
print(len(img_file))
# print(img_file)
if __name__ == "__main__":
    model = U_Net(num_channels, num_classes).to(device)
    model.load_state_dict(torch.load('UNet_weights_bilinear_weight.pth'))
    model.eval()
    with torch.no_grad():
        for i in range(len(img_file)):
            print(img_file[i])
            save_path=os.path.join("membrane1/result",os.path.basename(img_file[i])[0:-4]+".png")
            print(save_path)
            input = cv2.imread(img_file[i], cv2.IMREAD_COLOR)
            input = cv2.resize(input, size)
            original_img=input
            input = transforms.ToTensor()(input)
            input=transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(input)
            input = input.view((-1,)+input.shape)
            # print(input.shape)
            output = model(input.to(device))
            # print(output[:,1].shape)
            # print(output.shape)
            result=np.zeros((384,512,3))
            new_mask=torch.argmax(output,dim=1)
            print(new_mask.shape)
            for i in range(384):
                for j in range(512):
                    # print(output[0][:,i,j])
                    # print(new_mask[i,j,:])
                    if new_mask[0,i,j]==0:
                        result[i,j]=COLOR_DICT[0]
                    elif new_mask[0,i,j]==1 :
                        result[i, j] = COLOR_DICT[1]
                    else:
                        result[i, j] = COLOR_DICT[2]
            # result = cv2.resize(result, (512,512))
            cv2.namedWindow("test",cv2.WINDOW_NORMAL)
            # cv2.imshow("test",result)
            cv2.imwrite(save_path,result)
            # cv2.waitKey(0)
            # print(output[0])