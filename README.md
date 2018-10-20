# Pixel-wise-Image-Segmentation
Build the SegNet using VGG16 (Pytorch) <br>
<br>
First, prepare your train data in the folder and run "Train_segnet.py" <br>
During the train, validation images will be exported for every 10 steps <br>
Choose "compute_core = 0" to use CPU, "compute_core = 1" to use GPU <br>
The "ground_truth_label.py" automatically help you turn the "color label" from ground truth images into classes labels. <br>
We used the data from CamVid, the color codes are the colors in its ground truth images.<br>
<br>
After the training, the SegNet object will be saved by pickle. <br>
To use the train SegNet to make new segmentation, run "Test_net.py" with your prepared images<br>

![alt text](https://github.com/LinusCheng/Pixel-wise-Image-Segmentation/blob/master/Results.PNG)

