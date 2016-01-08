import os
import numpy as np
import matplotlib.pyplot as plt
import caffe

net_model_file = 'VGG_FACE_deploy.prototxt'
net_pretrained = 'VGG_FACE.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(0)

ChannelMean = [129.1863,104.7624,93.5940]
meanShape = (224,224)
mean = np.array([np.tile(ChannelMean[0],meanShape),np.tile(ChannelMean[1],meanShape),np.tile(ChannelMean[2],meanShape)])

VGG_Face_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(224, 224))


input_image = caffe.io.load_image('ak.png')
prediction = VGG_Face_Net.predict([input_image],oversample=False)
print len(prediction[0])