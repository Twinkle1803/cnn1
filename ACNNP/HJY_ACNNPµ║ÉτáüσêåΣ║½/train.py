import torchvision
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from Dataset_new import MyData
from NEW_CNN import HJY_AC_CNNP
import torch
from torch import nn
import numpy as np

import os
## Define training equipment
device = torch.device("cuda")


# 1. training data set

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
###  training data set
root_dir = './train_image'
test_dir = './ucid_gray'
###  Custom dataloader
train_dataset = MyData(root_dir,transforms_=dataset_transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
test_dataset = MyData(test_dir,transforms_=dataset_transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)

# 2. Build neural network
net = HJY_AC_CNNP()
net = net.to(device)
# 3. Define loss function
loss_fn = nn.MSELoss()
loss_fn = loss_fn.to(device)
# 4. Define optimizer
learning_rate = 1e-3
weight_decay = 1e-3
optimizer = torch.optim.Adam(net.parameters(),weight_decay=weight_decay, lr=learning_rate)

### Add tensorboard
#writer = SummaryWriter("./newlog")


### Set some parameters for training network
total_train_step = 0
total_test_step = 0
epoch = 300


#### dot as input

## Image preprocessing
dot = np.zeros((512,512), dtype=np.float32)
cross = np.zeros((512,512), dtype=np.float32)

### dot
### Image size must be a multiple of 4

for i in range(0,512):
    for j in range(0,512):
        if (i+j)%2 == 0:
            dot[i][j] = 1
        else:
            cross[i][j] = 1

print(dot)
print(cross)

cross = np.expand_dims(cross, axis=2)
dot = np.expand_dims(dot, axis=2)
dot = dataset_transform(dot)
cross = dataset_transform(cross)
cross = cross.to(device)
dot = dot.to(device)



### txt file records the training process
txt  = open('hjy_lognew.txt', 'a')



for i in range(epoch):
    print("_______________Training round {} starts(epoch)________".format(i+1))
    txt.write("\n_______________Training round {} starts(epoch)________".format(i+1))
    train_total_loss = 0
    ### Training begins
    net.train() ### Useful for specific layers such as dropout()
    for data in train_dataloader:
        img = data
        img = img.to(device)
        img_dot = torch.mul(img, dot)
        #print(img_dot)
        #print("hjyhjyhjyjhjyhjyjjhjy")
        img_cross = torch.mul(img, cross)
        #img_cross = img_cross[:,:,2:510,2:510]
        #print(img_dot)
        output = net(img_dot)
        #print(output)
        #print(img_cross)
        #print(img_cross.shape)
        #print(output.shape)
        loss = loss_fn(output, img_cross)
        ### Visualization
        #writer.add_scalar("train_loss", loss.item(), total_train_step)
        ### Backpropagation, parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        train_total_loss += loss.item()
        print("Loss at step {} is {}".format(total_train_step, loss.item()))
        txt.write("\nLoss at step {} is {}".format(total_train_step, loss.item()))
    print("train_total_loss = {}".format(train_total_loss))
    txt.write("\ntrain_total_loss = {}".format(train_total_loss))


    ### validation
    net.eval()
    total_test_loss = 0

    with torch.no_grad():
        for data in test_dataloader:
            img = data
            img = img.to(device)
            img_dot = torch.mul(img, dot)
            # print(img_dot)
            # print("hjyhjyhjyjhjyhjyjjhjy")
            img_cross = torch.mul(img, cross)
            #img_cross = img_cross[:,:,2:510,2:510]
            output = net(img_dot)
            loss = loss_fn(output,img_cross)
            total_test_loss += loss.item()
            total_test_step += 1
            #writer.add_scalar("test_loss", loss.item(), total_test_step)
    print("At epoch {}, the loss on the test dataset is {}".format(i+1,total_test_loss))
    txt.write("\nAt epoch {}, the loss on the test dataset is {}".format(i+1,total_test_loss))

    #writer.add_scalar("epoch_test_loss", total_test_loss, i+1)

    ## Save model
    if i % 5 == 0:
        torch.save(net, "Train_state{}.pth".format(i))

#writer.close()
txt.close()

###os.system("shutdown")

































