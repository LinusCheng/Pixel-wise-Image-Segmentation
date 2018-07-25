import numpy as np

def rgb2label(img_i):

    img_i_rgb = (img_i[0,:,:]+1)*1e6 + (img_i[1,:,:]+1)*1e3  + img_i[2,:,:]+1 +1e9
    
    color_list = np.array([
                    [64,128,64],
                    [192,0,128],
                    [0,128,192],
                    [0,128,64],
                    [128,0,0],
                    [64,0,128],
                    [64,0,192],
                    [192,128,64],
                    [192,192,128],
                    [64,64,128],
                    [128,0,192],
                    [192,0,64],
                    [128,128,64],
                    [192,0,192],
                    [128,64,64],
                    [64,192,128],
                    [64,64,0],
                    [128,64,128],
                    [128,128,192],
                    [0,0,192],
                    [192,128,128],
                    [128,128,128],
                    [64,128,192],
                    [0,0,64],
                    [0,64,64],
                    [192,64,128],
                    [128,128,0],
                    [192,128,192],
                    [64,0,64],
                    [192,192,0],
                    [0,0,0],
                    [64,192,0]])
    
    color_list +=1
    color_list_1D = color_list[:,0]*1e6 + color_list[:,1]*1e3 + color_list[:,2] +1e9
    for i in np.arange( len(color_list_1D)  ):
        img_i_rgb[img_i_rgb == color_list_1D[i]] = i
        
        
        
        
    return img_i_rgb.astype(np.uint8)




def label2rgb(img_out):

    color_list = np.array([
                [64,128,64],
                [192,0,128],
                [0,128,192],
                [0,128,64],
                [128,0,0],
                [64,0,128],
                [64,0,192],
                [192,128,64],
                [192,192,128],
                [64,64,128],
                [128,0,192],
                [192,0,64],
                [128,128,64],
                [192,0,192],
                [128,64,64],
                [64,192,128],
                [64,64,0],
                [128,64,128],
                [128,128,192],
                [0,0,192],
                [192,128,128],
                [128,128,128],
                [64,128,192],
                [0,0,64],
                [0,64,64],
                [192,64,128],
                [128,128,0],
                [192,128,192],
                [64,0,64],
                [192,192,0],
                [0,0,0],
                [64,192,0]])
    



    img_out_rgb = np.zeros((img_out.shape[0],img_out.shape[1],3))
    for i in np.arange(img_out.shape[0]):
        for j in np.arange(img_out.shape[1]):
            img_out_rgb[i,j,:] = color_list[int(img_out[i,j]),:]
            
            
            
#    img_zero = np.zeros_like(img_out)
#    img_out = np.concatenate(( img_out,img_zero,img_zero  ) ,axis =2)
#    img_out_rgb = np.zeros_like(img_out)
            
            
            
    return img_out_rgb.astype(np.uint8)







