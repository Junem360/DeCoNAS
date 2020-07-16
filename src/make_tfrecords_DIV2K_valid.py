import numpy as np
import os
import tensorflow as tf
from random import shuffle
import cv2
import glob
import src.utils_img as utils_img
from src.utils import DEFINE_integer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_integer("patch_size", 48, "patch size for valid data")
DEFINE_integer("upsample_size", 2, "rate of lr image size")
DEFINE_integer("patch_num", 500, "patch data_num of valid data")

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


sTrainDataPath = "./data/DIV2K_train_HR/*.png"
lTrainImgName = glob.glob(sTrainDataPath)


nFilesPerTfrecord = 10000
nPatchSize = FLAGS.patch_size*FLAGS.upsample_size
nStride = 120
nImg = len(lTrainImgName) 

nRot = 4 
nFlip = 2 
nPatch = 0 

lPatchNum = [] 
lImgSize = []
lPatchNum.append(0)

# for i in range(1): 
for i in range(len(lTrainImgName)):
    img = cv2.imread(lTrainImgName[i])
    # print(img.shape)
    height, width = img.shape[:2]
    lImgSize.append([height,width])
    patchNum = ((height-nPatchSize)//nStride + 1) * ((width-nPatchSize)//nStride + 1) * 8
    nPatch += patchNum
    lPatchNum.append(nPatch)

print("calc patch num finished!")
bIsShuffle = True
lPatchIdx = np.arange(nPatch) 
np.random.shuffle(lPatchIdx)
# lPatchIdx = [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]

j = 0
proc = 0
nTfrecord = nPatch // nFilesPerTfrecord
sTrainFileName = "./tfrecord/DIV2K_valid_x2_test/valid_x2_grad.tfrecord"
with tf.python_io.TFRecordWriter(sTrainFileName) as writer:
    while proc < FLAGS.patch_num:
        nPatchIdx = lPatchIdx[j]
        s = 0
        for k in range(len(lPatchNum)):
            if nPatchIdx - lPatchNum[k]< 0:
                s = k-1
                break
        nHeight = lImgSize[s][0]
        nWidth = lImgSize[s][1]
        nRow = (nHeight-nPatchSize)//nStride + 1
        nCol = (nWidth-nPatchSize)//nStride + 1
        h = (nPatchIdx-lPatchNum[s])//(nCol*nRot*nFlip)
        w = (nPatchIdx-lPatchNum[s]-nCol*nRot*nFlip*h)//(nRot*nFlip)
        r = (nPatchIdx-lPatchNum[s]-nCol*nRot*nFlip*h-nRot*nFlip*w)//nFlip
        f = nPatchIdx-lPatchNum[s]-nCol*nRot*nFlip*h-nRot*nFlip*w-nFlip*r

        sImgName = lTrainImgName[s]
        ImgData = cv2.imread(sImgName).astype(np.float32)/255
        height, width, channel = ImgData.shape
        ImgPatchData = ImgData[h*nStride:h*nStride+nPatchSize, w*nStride:w*nStride+nPatchSize,::]
        if r > 0:
            for _ in range(r):
                ImgPatchData = np.rot90(ImgPatchData)
        if f > 0 :
            ImgPatchData = ImgPatchData[:,::-1,:]

        if utils_img.gradients(ImgPatchData*255) > 51.065:
            # LRImgPatchData = utils_img.imresize(utils_img.imresize(ImgPatchData, 0.5), 2).astype(np.float32)
            LRImgPatchData = utils_img.imresize(ImgPatchData, 0.5)
            LRImgPatchData = (np.round(np.clip(LRImgPatchData * 255., 0., 255.)) / 255).astype(np.float32)

            # LRImgPatchData = utils_img.imresize(LRImgPatchData, 2)
            # LRImgPatchData = (np.round(np.clip(LRImgPatchData * 255., 0., 255.)) / 255).astype(np.float32)

            # cv2.imwrite('gt_%02d.png' % i, ImgPatchData*255)
            feature = {'train/input_LR': _bytes_feature(LRImgPatchData.tostring()),
                       'train/gt_HR': _bytes_feature(ImgPatchData.tostring())}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            if j % 50 == 0:
                print('Train data : {}/{}  is generated!'.format(j, nPatch))
            # cv2.imshow('patch', ImgPatchData)
            # cv2.imshow('LRpatch', LRImgPatchData)
            # cv2.waitKey(0)
            j += 1
            proc += 1
        else:
            if j % 50 == 0:
                print("patch number {}'s gradient is low!".format(j))
            # cv2.imshow('patch', ImgPatchData)
            # cv2.waitKey(0)
            j += 1
            continue





