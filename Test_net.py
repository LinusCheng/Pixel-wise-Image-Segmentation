import numpy as np
from PIL import Image
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.autograd import Variable


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
            
def load_test_photos(dataset_name):
    file_list = list(listdir_nohidden(dataset_name))
    X_np_list =[]
    transform = torchvision.transforms.Resize(180,interpolation=Image.NEAREST)

    for input_name in file_list:
        img = transform(Image.open(dataset_name+input_name))
#        img  = Image.open(dataset_name+input_name)
        img_i = np.swapaxes(np.swapaxes(np.array(img),0,2 ),1,2)
        X_np_list.append(img_i)
    X_np = np.asarray(X_np_list)
    return X_np
    
X_np = load_test_photos('Test_data/')

    
print("Photos loaded")




""" Load the Trained SegNet """
import pickle
pickle_in  = open("segnet_101_8463iter.pickle","rb")
segnet = pickle.load(pickle_in)
pickle_in.close()


segnet.type(dtype)


for ii in np.arange(X_np.shape[0]):

    X_np_one = X_np[ii,:,:,:].reshape((1,X_np.shape[1],X_np.shape[2],X_np.shape[3]))
    X_test = Variable(torch.from_numpy(X_np_one).type(dtype))

    H    = segnet(X_test)

    
    H_out_test = (H.data).cpu().numpy()[0,:,:,:]
    H_out_test = np.argmax(H_out_test,axis =0).T
    img_out_rgb = gtl.label2rgb(H_out_test) 
    img_out_rgb = np.swapaxes(img_out_rgb,0,1)
    image = Image.fromarray(img_out_rgb.astype('uint8'), 'RGB')
    image.save("Test_pred_data/pred_" + str(ii) + ".png", "png")


print("completed")

