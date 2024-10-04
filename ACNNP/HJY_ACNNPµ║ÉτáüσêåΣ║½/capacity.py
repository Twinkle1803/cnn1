
import torchvision
from torch.utils.data import DataLoader
from NEW_CNN import HJY_AC_CNNP
from Dataset_new import MyData
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
a = 0
b = 0
#torch.save(model, "test.pth")
with torch.no_grad():
    for data in test_dataloader:
        print(ii)
        img, img_self, name = data
        print(name)
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
        ## Model prediction
        predicted_image = model(img_dot)
        #print(predicted_image.shape)
        #print(predicted_image)
        ## Dimension transformation
        predicted_image = torch.squeeze(predicted_image)
        #print(predicted_image.shape)
        #print(predicted_image)
        ## Data type conversion
        predicted_image = predicted_image.cpu().numpy()
        predicted_image = np.around(predicted_image)
        predicted_image[predicted_image<0] = 0
        #print(predicted_image)
        ## Convert to uint8
        predicted_image = predicted_image.astype(np.uint8)
#########################################################################################
        ## Initialize some data for data embedding
        label_set = {"{}".format(i): i - i for i in range(9)}
        #print(label_set)
        label_matrix = np.zeros((512, 512), dtype=np.uint8)
        #print(label_matrix)
        ## Simple Huffman coding
        ## unnecessary code = ["11111", "11110", "1110", "1101", "1100", "101", "100", "01", "00"]
        huff_code = dict()
        ### img_self is the original image, predicted_image is the predicted cross_iamge
        ### Based on the predicted cross-image, in cross_imge, calculate the MSB, Huffman coding, and embed the data.
        ## First, count the number of embedding digits and perform Huffman coding
        ## The default image size processed is 512*512
        for i in range(512):
            for j in range(512):
                if (i+j) % 2 == 1:
                    original_pixel = img_self[i][j]
                    predicted_pixel = predicted_image[i][j]
                    #print("this is orginal {} and this is predicted {} ".format(original_pixel, predicted_pixel))
                    #print()
                    ## Convert to binary
                    original_bin = '{0:08b}'.format(original_pixel)
                    predicted_bin ='{0:08b}'.format(predicted_pixel)
                    #print("this is original {} and this is the prredicted {}".format(original_bin, predicted_bin))
                    #print(type(original_bin))
                    #print(type(predicted_bin))
                    label_index = 0
                    while label_index < 8:
                        if original_bin[label_index] == predicted_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    #print(label_index)
                    label_set[str(label_index)] += 1
                    label_matrix[i][j] = label_index
        #print(label_set)
        sorted_label_set = sorted(label_set.items(), key=lambda x: x[1])
        #print(sorted_label_set)
        index = 0
        """
        for label in sorted_label_set:
            huff_code[label[0]] = code[index]
            index += 1
        """
        tree2 = HuffmanTree(sorted_label_set)
        huff_code = tree2.get_code()
        #print(huff_code)
        #print(label_matrix)
        ######## Computing capacity ###############
        cross_total_capacity = 0
        cross_aux_information = 0
        for key in label_set:
            if key == '8':
                cross_total_capacity += 8 * label_set[key]
                cross_aux_information += len(huff_code[key]) * label_set[key]
            else:
                cross_total_capacity += (int(key) + 1) * label_set[key]
                cross_aux_information += len(huff_code[key]) * label_set[key]
        capacity_cross = cross_total_capacity - cross_aux_information
        print("Cross effective embedding capacity is {} bits".format(capacity_cross))
        print("Cross total embedding capacity is {} bits".format(cross_total_capacity))
        print("Cross auxiliary information size is {} bits".format(cross_aux_information))
        """
        print("总的嵌入容量为 {} bit".format(total_capacity))
        print("辅助信息的大小为 {} bit".format(aux_information))
        print("有效嵌入容量为 {} bit".format(total_capacity - aux_information))
        """

        ###################  dotdotdotdotdotdotdotdot   ##############
        #print("start dot processing !!!!!!!!!!!!!!")
        # print(img_dot)
        ######## top #####################

        img_dot = img_self * hjy1

        ## Initialize some data
        #print("this is the top predictor")
        label_set = {"{}".format(i): i - i for i in range(9)}
        label_matrix = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()

        ##Simple top edge median prediction
        ## The first column, first row, and last column are not used
        for i in range(1, 512):
            for j in range(1, 511):
                if (i + j) % 2 == 0:
                    ## Data will overflow
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i - 1][j - 1]) + np.uint16(img_dot[i - 1][j + 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_set[str(label_index)] += 1
                    label_matrix[i][j] = label_index
        #print(label_set)
        # print(label_set)
        sorted_label_set = sorted(label_set.items(), key=lambda x: x[1])
        #print(sorted_label_set)
        index = 0
        """
        for label in sorted_label_set:
            huff_code[label[0]] = code[index]
            index += 1
        """
        tree2 = HuffmanTree(sorted_label_set)
        huff_code = tree2.get_code()
        #print(huff_code)
        # print(label_matrix)
        ######## Computing capacity ###############
        total_capacity = 0
        aux_information = 0
        for key in label_set:
            if key == '8':
                total_capacity += 8 * label_set[key]
                aux_information += len(huff_code[key]) * label_set[key]
            else:
                total_capacity += (int(key) + 1) * label_set[key]
                aux_information += len(huff_code[key]) * label_set[key]
        temp_capacity1 = total_capacity - aux_information
        print("1 effective embedding capacity is {} bits".format(temp_capacity1))
        print("1 total embedding capacity is {} bits".format(total_capacity))
        print("1 auxiliary information size is {} bits".format(aux_information))



        ######## bottom #####################
        #print("this is the botttom predictor")
        ## Initialize some data
        label_set_bottom = {"{}".format(i): i - i for i in range(9)}
        label_matrix_bottom = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()

        ## Simple top edge median prediction
        ## left, right , bottom notNeedede
        for i in range(510, -1, -1):
            for j in range(510, 0, -1):
                if (i + j) % 2 == 0:
                    ## Data will overflow
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i + 1][j + 1]) + np.uint16(img_dot[i + 1][j - 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_set_bottom[str(label_index)] += 1
                    label_matrix_bottom[i][j] = label_index
        #print(label_set_bottom)
        # print(label_set)
        sorted_label_set_bottom = sorted(label_set_bottom.items(), key=lambda x: x[1])
        #print(sorted_label_set_bottom)
        tree3 = HuffmanTree(sorted_label_set_bottom)
        huff_code_bottom = tree3.get_code()
        #print(huff_code_bottom)
        # print(label_matrix)
        ######## Computing capacity ###############
        total_capacity1 = 0
        aux_information1 = 0
        for key in label_set_bottom:
            if key == '8':
                total_capacity1 += 8 * label_set_bottom[key]
                aux_information1 += len(huff_code_bottom[key]) * label_set_bottom[key]
            else:
                total_capacity1 += (int(key) + 1) * label_set_bottom[key]
                aux_information1 += len(huff_code_bottom[key]) * label_set_bottom[key]

        temp_capacity2 = total_capacity1 - aux_information1
        print("2 effective embedding capacity is {} bits".format(temp_capacity2))
        print("2 total embedding capacity is {} bits".format(total_capacity1))
        print("2 auxiliary information size is {} bits".format(aux_information1))
        ######## bottom #####################
        #print("this is the left predictor")
        ## Initialize some data
        label_setr = {"{}".format(i): i - i for i in range(9)}
        label_matrixr = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()

        ## Simple top edge median prediction
        ##  left, top , bottom notNeeded
        for i in range(1, 511):
            for j in range(1, 512):
                if (i + j) % 2 == 0:
                    ## Data will overflow
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i - 1][j - 1]) + np.uint16(img_dot[i + 1][j - 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_setr[str(label_index)] += 1
                    label_matrixr[i][j] = label_index

        #print(label_setr)
        # print(label_set)
        sorted_label_setr = sorted(label_setr.items(), key=lambda x: x[1])
        #print(sorted_label_setr)
        tree3 = HuffmanTree(sorted_label_setr)
        huff_coder = tree3.get_code()
        #print(huff_coder)
        # print(label_matrix)
        ######## Computing capacity ###############
        total_capacityr = 0
        aux_informationr = 0
        for key in label_setr:
            if key == '8':
                total_capacityr += 8 * label_setr[key]
                aux_informationr += len(huff_coder[key]) * label_setr[key]
            else:
                total_capacityr += (int(key) + 1) * label_setr[key]
                aux_informationr += len(huff_coder[key]) * label_setr[key]


        temp_capacity3 = total_capacityr - aux_informationr
        print("3 effective embedding capacity is {} bits".format(temp_capacity3))
        print("3 total embedding capacity is {} bits".format(total_capacityr))
        print("3 auxiliary information size is {} bits".format(aux_informationr))

        ######## bottom #####################
        #print("this is the right predictor")
        ## Initialize some data
        label_setl = {"{}".format(i): i - i for i in range(9)}
        label_matrixl = np.zeros((512, 512), dtype=np.uint8)
        ###huff_code = dict()
        ## Simple top edge median prediction
        ##  right, top , bottom notNeeded
        for i in range(1, 511):
            for j in range(510, -1, -1):
                if (i + j) % 2 == 0:
                    ## Data will overflow
                    # a = np.uint16(img_dot[i-1][j-1])
                    temp1 = (np.uint16(img_dot[i - 1][j - 1]) + np.uint16(img_dot[i + 1][j - 1])) // 2
                    temp2 = img_dot[i][j]
                    temp1_bin = '{0:08b}'.format(temp1)
                    temp2_bin = '{0:08b}'.format(temp2)
                    label_index = 0
                    while label_index < 8:
                        if temp1_bin[label_index] == temp2_bin[label_index]:
                            label_index += 1
                        else:
                            break
                    # print(label_index)
                    label_setl[str(label_index)] += 1
                    label_matrixl[i][j] = label_index
        #print(label_setl)
        # print(label_set)
        sorted_label_setl = sorted(label_setl.items(), key=lambda x: x[1])
        #print(sorted_label_setl)
        tree3 = HuffmanTree(sorted_label_setl)
        huff_codel = tree3.get_code()
        #print(huff_codel)
        # print(label_matrix)
        ######## Computing capacity ###############
        total_capacityl = 0
        aux_informationl = 0
        for key in label_setl:
            if key == '8':
                total_capacityl += 8 * label_setl[key]
                aux_informationl += len(huff_codel[key]) * label_setl[key]
            else:
                total_capacityl += (int(key) + 1) * label_setl[key]
                aux_informationl += len(huff_codel[key]) * label_setl[key]

        temp_capacity4 = total_capacityl - aux_informationl
        print("4 effective embedding capacity is {} bits".format(temp_capacity4))
        print("4 total embedding capacity is {} bits".format(total_capacityl))
        print("4 auxiliary information size is {} bits".format(aux_informationl))


        dot_capacity = max(temp_capacity1, temp_capacity2, temp_capacity3, temp_capacity4)
        print("Dot effective embedding capacity is {} bits".format(dot_capacity))
        capacity = capacity_cross + dot_capacity
        print("Effective embedding capacity is {} bits".format(capacity))
        print("Effective embedding capacity is {} bpp".format(capacity / 512 / 512))


        if ii == 0:
            capacity_best = capacity / 512 / 512
            capacity_worse = capacity / 512 / 512


        if (capacity/512/512) >= capacity_best:
            capacity_best = capacity/512/512
        if (capacity/512/512) <= capacity_worse:
            capacity_worse = capacity/512/512
        ii += 1
        capacity_average += capacity/512/512
print("the best is {}".format(capacity_best))
print("the worse is {}".format(capacity_worse))
print("the average is {}".format(capacity_average/10000))















































