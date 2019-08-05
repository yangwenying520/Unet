import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np

# background = [255,255,255]
# boundary= [0,0,0]
# COLOR_DICT =[background, boundary]
#
# def search_file(data_path, target):  # 寻找目录及其子文件下所有指定扩展名的文件
#     video_list = []
#     for dirpath, dirnames, filenames in os.walk(data_path):
#         for f in filenames:
#             ext = os.path.splitext(f)[1]
#             if ext in target:
#                 video_list.append(os.path.join(dirpath, f))
#     return video_list
#
# class My_Dataset(Dataset):
#     def __init__(self,root,num_classes,size,transform=None,mask_transform=None):
#         super(My_Dataset, self).__init__()
#         self.root=root
#         self.size=size
#         self.transform=transform
#         self.mask_transform=mask_transform
#         self.image_dir=os.path.join(self.root,"image")
#         self.mask_dir=os.path.join(self.root,"label")
#         img_file=search_file(self.image_dir,[".png"])
#         label_file=search_file(self.mask_dir,[".pang"])
#         self.sample={"img":img_file,"mask":label_file}
#         # print(self.sample["img"])
#         self.num_classes=num_classes
#     def __len__(self):
#         return len(self.sample["img"])
#
#     def __getitem__(self, item):#__getitem__的重写
#         original=cv2.imread(self.sample["img"][item],cv2.IMREAD_COLOR)
#         original=cv2.resize(original,self.size)
#         # print(os.path.join(self.mask_dir, os.path.basename(self.sample["img"][item])))
#         mask=cv2.imread(os.path.join(self.mask_dir,os.path.basename(self.sample["img"][item])),cv2.IMREAD_GRAYSCALE)
#         mask = cv2.resize(mask, self.size)
#         # print(mask.shape)
#         # new_mask=np.zeros(mask.shape+(self.num_classes,))
#         new_mask=np.zeros(mask.shape)
#         new_mask[mask == 255] = 0
#         new_mask[mask == 0] = 1
#         if self.transform:
#             img=self.transform(original)
#         # if self.mask_transform:
#         #     new_mask=self.mask_transform(new_mask)
#         new_mask=torch.from_numpy(new_mask)
#         """transforms.ToTensor()  Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#         [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0],but NLLLoss2d  Input: (N,C,H,W)  C 类的数量
#         Target: (N,H,W) where each value is 0 <= targets[i] <= C-1,so we use torch.from_numpy to convert numpy to tensor"""
#         return img,new_mask
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np

background = [0,0,0]
boundary= [0,255,0]
boundary1= [0,0,255]
COLOR_DICT =[background, boundary,boundary1]

def search_file(data_path, target):  # 寻找目录及其子文件下所有指定扩展名的文件
    video_list = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for f in filenames:
            ext = os.path.splitext(f)[1]
            if ext in target:
                video_list.append(os.path.join(dirpath, f))
    return video_list

class My_Dataset(Dataset):
    def __init__(self,root,num_classes,size,transform=None,mask_transform=None):
        super(My_Dataset, self).__init__()
        self.root=root
        self.size=size
        self.transform=transform
        self.mask_transform=mask_transform
        self.image_dir=os.path.join(self.root,"image")
        self.mask_dir=os.path.join(self.root,"label")
        img_file=search_file(self.image_dir,[".jpg"])
        label_file=search_file(self.mask_dir,[".pang"])
        self.sample={"img":img_file,"mask":label_file}
        # print(self.sample["img"])
        self.num_classes=num_classes
    def __len__(self):
        return len(self.sample["img"])

    def __getitem__(self, item):#__getitem__的重写
        original=cv2.imread(self.sample["img"][item],cv2.IMREAD_COLOR)
        original=cv2.resize(original,self.size)
        # print(os.path.join(self.mask_dir, os.path.basename(self.sample["img"][item])))
        # print(os.path.basename(self.sample["img"][item])[0:-4])
        mask=cv2.imread(os.path.join(self.mask_dir,os.path.basename(self.sample["img"][item])[0:-4]+".png"),cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.size,cv2.INTER_NEAREST)
        # print(mask)
        # print(mask.shape)
        # new_mask=np.zeros(mask.shape+(self.num_classes,))
        new_mask=np.zeros(mask.shape[0:-1])
        for i in range(384):
            for j in range(512):
                # print(mask[i,j])
                # print(mask[383, 511])
                if mask[i,j,0]!=0:
                    new_mask[i,j]=0
                if mask[i, j, 1] != 0:
                    new_mask[i,j]=1
                if mask[i, j, 2] != 0:
                    new_mask[i, j] = 2
        # new_mask[mask == [0,0,0]] = 0
        # new_mask[mask == [0,255,0]] = 1
        # new_mask[mask == [0, 0, 128]] = 2
        if self.transform:
            img=self.transform(original)
        # if self.mask_transform:
        #     new_mask=self.mask_transform(new_mask)
        new_mask=torch.from_numpy(new_mask)
        """transforms.ToTensor()  Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0],but NLLLoss2d  Input: (N,C,H,W)  C 类的数量
        Target: (N,H,W) where each value is 0 <= targets[i] <= C-1,so we use torch.from_numpy to convert numpy to tensor"""
        return img,new_mask


if __name__=="__main__":
    original = cv2.imread("../data/membrane/train/image/0.png",cv2.IMREAD_ANYCOLOR)
    mask = cv2.imread("../data/membrane/train/label/2007_000032.png", cv2.IMREAD_ANYDEPTH)
    print(mask.shape)
    # for i in range(512):
    #     print(mask[i,i])
    print(mask.shape)
    new_mask = np.zeros(mask.shape + (2,))
    new_mask[mask==255,0]=1
    new_mask[mask == 0, 1] = 1
    for i in range(281):
        for j in range(500):
            print(mask[i,j])
    print(new_mask)
