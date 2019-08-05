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
from torch import  optim
from torchvision import transforms
import torch.nn as nn
import os
from network import U_Net
from utils import My_Dataset
from tensorboardX import SummaryWriter
from utils import MulticlassDiceLoss
from utils import loss_weight

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
num_classes = 3
num_channels = 3
batch_size = 4
size=(512,384)
num_epochs=100
root = "./membrane/train"
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight=loss_weight()
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# mask只需要转换为tensor
mask_transforms = transforms.ToTensor()


def train_model(model, criterion, optimizer, dataload, model_graph=None,num_epochs=10):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('--' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0

        for x, y in dataload:
            step += 1
            outputs_weight=weight.class_weight(y.numpy())
            # weight.distance_weight(y.numpy())
            outputs_weight=torch.from_numpy(outputs_weight).to(device)
            inputs = x.to(device)
            labels = y.to(device)
            labels=labels.long()
            # print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # outputs=(outputs.double()).mul(outputs_weight.double())
            # print(outputs)
            # outputs.float()
            # print(outputs.shape,labels.shape)
            # outputs=outputs.long()
            # labels=labels.long()
            loss = criterion(outputs,labels)
            print(loss.requires_grad)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        model_graph.add_scalar("train",epoch_loss,epoch )
    torch.save(model.state_dict(), 'UNet_weights_bilinear_weight.pth')


def train():
    model = U_Net(n_channels=num_channels, n_classes=num_classes).to(device)
    model_graph = SummaryWriter(comment="UNet")
    # input_c = torch.rand(1, 3, 256, 256)
    # model_graph.add_graph(model, (input_c.to(device),))
    model.train()
    # criterion = nn.BCELoss()
    criterion = nn.NLLLoss2d()
    # criterion=MulticlassDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = My_Dataset(root, num_classes, size, transform=img_transforms, mask_transform=mask_transforms)
    data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, data_loaders,model_graph=model_graph,num_epochs=num_epochs)
    model_graph.close()


if __name__ == "__main__":
    train()
    # torch.cuda.empty_cache()
