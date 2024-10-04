
import torchvision
from torch.utils.data import DataLoader
from NEW_CNN  import HJY_AC_CNNP
from Dataset_new_two import MyData
import torch
from torch import nn
import numpy as np
import cv2
############## Huffman coding implementation #################
## Node class
class Node(object):
    def __init__(self,name = None, value = None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None
## Huffman trees
class HuffmanTree(object):
    ## According to the idea of ​​Huffman Tree, Huffman tree is built in reverse based on leaf nodes.
    def __init__(self, char_weights):
        ## Generate leaf nodes based on frequency of characters
        self.a = [Node(part[0], part[1]) for part in char_weights]
        while len(self.a) != 1:
            self.a.sort(key=lambda node:node._value,reverse=True)
            c = Node(value=(self.a[-1]._value + self.a[-2]._value))
            c._left = self.a.pop(-1)
            c._right = self.a.pop(-1)
            self.a.append(c)
        self.root = self.a[0]
        self.b = list(range(10))
        self.huffman_code = dict()
    ## Recursive Thought Generating Coding
    def pre(self,tree,length):
        node = tree
        if (not node):
            return
        elif node._name:
            x = ""
            for i in range(length):
                x += str(self.b[i])
            #print(x)
            self.huffman_code[node._name] = x
            #print(self.huffman_code)
            return
        self.b[length] = 0
        self.pre(node._left,length+1)
        self.b[length] = 1
        self.pre(node._right,length+1)
    def get_code(self):
        hjy = self.pre(self.root,0)
        return self.huffman_code


#### Define capacity !!!!!!!!
capacity_best = 0
capacity_average = 0
capacity_worse = 0

## Define test equipment
# 1. training data set
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
###  training data set
test_dir = './test'
###  Custom dataloader
test_dataset = MyData(test_dir,transforms_=dataset_transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
## 4. Image preprocessing
dot = np.zeros((512,512), dtype=np.float32)
cross = np.zeros((512,512),dtype=np.float32)
hjy1 = np.zeros((512,512),dtype=np.uint8)
for i in range(512):
    for j in range(512):
        if (i + j)%2 == 0:
            dot[i][j] = 1
            hjy1[i][j] = 1
        if (i + j)%2 == 1:
            cross[i][j] = 1
            hjy1[i][j] = 1

cross = np.expand_dims(cross, axis=2)
dot = np.expand_dims(dot, axis=2)
dot = dataset_transform(dot)
cross = dataset_transform(cross)
### secret_img is the final saved enigmatic image
secret_img = np.zeros((512,512),dtype=np.uint8)
### 5. Load model
file_name = './Train_state240.pth'
model = torch.load(file_name, map_location='cpu')
ii = 0
#torch.save(model, "test.pth")
with torch.no_grad():
    for data in test_dataloader:
        print(ii)
        img, img_self = data
        #print(img)
        #print(img_self.shape)
        #print(img_self)
        img_self = torch.squeeze(img_self)
        #print(type(img_self))
        img_self = img_self.cpu().numpy()
        #print(img_self.shape)
        #print("this is the original image")
        #print(img_self)
        img_dot = torch.mul(img, dot)
        img_cross = torch.mul(img, cross)
        #print("this is  !!!!!!!!!!!!!")
        #print(img_cross.shape)
        #print(type(img_cross))
        ## 模型预测
        predicted_image = model(img_dot)
        #print(predicted_image.shape)
        #print(predicted_image)
        ## 维度变换
        predicted_image = torch.squeeze(predicted_image)
        #print(predicted_image.shape)
        #print(predicted_image)
        ## 数据类型转换
        predicted_image = predicted_image.cpu().numpy()
        predicted_image = np.around(predicted_image)
        predicted_image[predicted_image<0] = 0
        #print(predicted_image)
        ## Convert to uint8
        predicted_image = predicted_image.astype(np.uint8)
        ii += 1
        cv2.imwrite("predicted{}.png".format(ii), predicted_image)
        















































