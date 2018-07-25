import numpy as np
from PIL import Image
import os

#from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision


import torch.utils.data as data

import time

import segnet_class as sg
import ground_truth_label as gtl


os.chdir('E:\\Dropbox\\MobileProjects\\211A\\SegNet')

""" Hyper params """

compute_core = 1
num_label    = 32

LR    = 0.01
mmtm  = 0.5


EPOCH   = 1



""" CPU or GPU """
if   compute_core == 0:
    dtype  = torch.FloatTensor
    dtype2 = torch.LongTensor 
elif compute_core == 1:
    dtype  = torch.cuda.FloatTensor 
    dtype2 = torch.cuda.LongTensor  
else:
    print("Choose compute_core again")


""" Load photos """


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
#def listdir_nohidden(path):
#    return glob.glob(os.path.join(path, '*'))
            
def load_photos(dataset_name):
    file_list = list(listdir_nohidden(dataset_name))
    X_np_list =[]
    Y_np_list =[]
    transform = torchvision.transforms.Resize(180,interpolation=Image.NEAREST)

    for input_name in file_list:
        
        img = transform(Image.open(dataset_name+input_name))
#        img  = Image.open(dataset_name+input_name)
        img_i = np.swapaxes(np.swapaxes(np.array(img),0,2 ),1,2)
        if input_name[-5] == 'L':
            img_i = gtl.rgb2label(img_i)
            Y_np_list.append(img_i)
        else:    
            X_np_list.append(img_i)
    X_np = np.asarray(X_np_list)
    Y_np = np.asarray(Y_np_list)
    return X_np,Y_np



"""  If you have your own photos, uncomment these lines """
#X_np,Y_np = load_photos('train_data/CamSeq01/')
#X_np,Y_np = load_photos('train_data/CamVid701/')

""" Then you can save the extracted numpy arrays """
#np.savez('X_Y_CamSeq01', X_np =X_np ,Y_np =Y_np) 
#np.savez('X_Y_CamVid701', X_np =X_np ,Y_np =Y_np)


""" We can load the saved numpy arrays """
X_np = np.load('X_Y_CamSeq01.npz')['X_np']
Y_np = np.load('X_Y_CamSeq01.npz')['Y_np']

#X_np = np.load('X_Y_CamVid701.npz')['X_np']
#Y_np = np.load('X_Y_CamVid701.npz')['Y_np']



""" Choosing regions of train and test photos """
#X_train_np = X_np[0:95,:,:,:]
#y_train_np = Y_np[0:95,:,:]
#
#X_test_np = X_np[95:,:,:,:]
#Y_test_np = Y_np[95:,:,:]



#X_train = torch.from_numpy(X_train_np).type(dtype)
#Y_train = torch.from_numpy(y_train_np).type(dtype)
#
#X_test = Variable (torch.from_numpy(X_test_np).type(dtype), requires_grad=True)
#Y_test = torch.from_numpy(Y_test_np).type(dtype)


X_np_one = X_np[0,:,:,:].reshape((1,X_np.shape[1],X_np.shape[2],X_np.shape[3]))
X_try = Variable(torch.from_numpy(X_np_one).type(dtype), requires_grad=True)



X_train = torch.from_numpy(X_np[1:,:,:,:]).type(dtype)
Y_train = torch.from_numpy(Y_np[1:,:,:]).type(dtype)



del X_np,Y_np  #, X_train_np,y_train_np, X_test_np, Y_test_np



""" Loading SegNet """
segnet = sg.SegNet(num_label)
segnet.type(dtype)




optimizer = torch.optim.Adam(segnet.parameters(),lr=LR)  
#optimizer = torch.optim.SGD(segnet.parameters() ,lr=LR, momentum=mmtm)

#loss_func = nn.CrossEntropyLoss().type(dtype)     

loss_func = nn.CrossEntropyLoss() 




""" ++++++ Train ++++++ """

batch_size =10

TrainData = data.TensorDataset(X_train,Y_train)
loader    = data.DataLoader(
    dataset=TrainData,      
    batch_size=batch_size,      
    shuffle=True,            
#    num_workers=2,             
)


print("Start iteration")


ii=0
for epoch in range(10000):
    print("epoch:",epoch)
    
    
    for step,(x,y) in enumerate(loader):
        
#        print("step = ",step,", ii  = ",ii)

        
        batch_X = Variable(x,requires_grad=True)             # batch x
        batch_Y = Variable(y.type(dtype2))                   # batch y
    
        H    = segnet(batch_X)
        loss = loss_func(H,batch_Y)
        # eat long tenor
#        print("optimizer.zero_grad()")
        optimizer.zero_grad()           # clear gradients for this training step
#        print("loss.backward()")
#        t1 = time.time()       
        loss.backward()                 # backpropagation, compute gradients   
#        t2 = time.time()
#        print(t2-t1)
#        print("optimizer.step()")
        optimizer.step()                # apply gradients

        if ii %10 ==0:
            
            H = segnet(X_try)
            H_out_test = (H.data).cpu().numpy()[0,:,:,:]
            H_out_test = np.argmax(H_out_test,axis =0).T
            img_out_rgb = gtl.label2rgb(H_out_test) 
            img_out_rgb = np.swapaxes(img_out_rgb,0,1)
            image = Image.fromarray(img_out_rgb.astype('uint8'), 'RGB')
            
            image.save("fig_SGD/SGD step_" + str(ii) + ".png", "png")
            print("image saved")
            
        ii +=1

##        if step % 500 == 0:
#        
#        print("Y_test_pred....")
#        
##        Y_test_pred = segnet(X_test)
#        Y_test_pred = segnet(X_try)
#
#        
#        print("torch.max")
#        Y_test_pred = torch.max(Y_test_pred, 1)[1].data.squeeze()
#        # torch.max output longtensor 
#        accuracy = sum(Y_test_pred == X_try) / float(X_try.size(0))
#        print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % accuracy)





print("comepleted")


""" Save the weights in SegNet as an object """

import pickle
pickle_out  = open("segnet_101_iter.pickle","wb")
pickle.dump(segnet,pickle_out)
pickle_out.close()

