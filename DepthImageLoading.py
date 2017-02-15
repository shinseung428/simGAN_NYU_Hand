import scipy.io
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from collections import namedtuple
import math
import time

DepthFrame = namedtuple('DepthFrame',['dpt','gtorig','gtcrop','T','gt3Dorig','gt3Dcrop','com','fileName','subSeqName'])

class DepthMapHandle(object):
    
    def __init__(self, fx, fy, ux, uy, cube_size, img_size):
        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy
        self.cube_size = cube_size
        self.img_size = img_size

    def loadDepthMap(self, filename):
        with open(filename) as f:
            img = Image.open(filename)
            assert len(img.getbands()) == 3
            r, g, b = img.split()
            r = np.asarray(r,np.int32)
            g = np.asarray(g,np.int32)
            b = np.asarray(b,np.int32)
            dpt = np.bitwise_or(np.left_shift(g,8),b)
            imgdata = np.asarray(dpt,np.float32)

        return imgdata

    def cvLoadDepthMap(self, filename):
        img = cv2.imread(filename)
        #cv2.imshow("ori", img)
        imgdata = np.asarray(img[:,:,0] + img[:,:,1]*256)
        return np.asarray(imgdata)
       

    def saveDepthMap(self, dpt):
        lower = np.bitwise_and(dpt, 255)
        higher = np.bitwise_and(np.right_shift(dpt, 8), 255)

        dpt3ch = np.zeros((self.img_size,self.img_size,3), np.uint8)
        dpt3ch[:,:,0] = lower;
        dpt3ch[:,:,1] = higher;
        dpt3ch[:,:,2] = 0;
        return dpt3ch
    
    def saveNormalizedDepthMap(self, dpt, com):

        mask = np.bitwise_and(dpt>850 , dpt != 0)
        #dpt[mask] = 0
        #mask_ = (dpt!=0)
        #dpt = dpt*mask_
        #cv2.imshow("dpt", dpt*255)

        dpt32 = np.zeros((self.img_size,self.img_size), np.float32)

        min_ = com[2] - self.cube_size/2
        max_ = com[2] + self.cube_size/2

        dpt32 = ((dpt - min_) / (max_ - min_)).reshape(self.img_size, self.img_size)
        #dpt32 = dpt32/np.max(dpt)
        
        return dpt32#*(self.cube_size / 2)


    def cropImage(self, image, com):

        u, v, d = com
        zstart = d - self.cube_size / 2.
        zend = d + self.cube_size / 2.
        xstart = int(math.floor((u * d / self.fx - self.cube_size / 2.) / d * self.fx))
        xend = int(math.floor((u * d / self.fx + self.cube_size / 2.) / d * self.fx))
        ystart = int(math.floor((v * d / self.fy - self.cube_size / 2.) / d * self.fy))
        yend = int(math.floor((v * d / self.fy + self.cube_size / 2.) / d * self.fy))
        #print xstart, xend, ystart, yend, zstart, zend
        cropped = image[max(ystart, 0):min(yend, image.shape[0]), max(xstart, 0):min(xend, image.shape[1])].copy()
        cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, image.shape[0])), 
                                    (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, image.shape[1]))), mode='constant', constant_values=0)

        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)

        cropped[msk1] = zstart
        cropped[msk2] = zend
        #print zstart, zend, d

        dsize = (self.img_size, self.img_size)
        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        rz = cv2.resize(cropped, sz)
        ret = np.ones(dsize, np.float32) * zend
        #ret = np.zeros(dsize, np.uint16)

        xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
        xend = int(xstart + rz.shape[1])
        ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        #print xstart, xend, ystart, yend
        return ret 


if __name__ == '__main__':
    data_path = '/home/vtouch-simulation/global_data/nyu_hand/dataset/train'
    #data_path = 'D://Data/Hand/NYU/dataset/test_2'
    numTotalJoint = 36
    jointIndex =[0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
    #jointIndex =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    numJoints = len(jointIndex)

    trainlabels = '/home/vtouch-simulation/global_data/nyu_hand/dataset/train/joint_data.mat'
    mat = scipy.io.loadmat(trainlabels)
    
    #testlabels = 'D://Data/Hand/NYU/dataset/test_2/joint_data.mat'
    #mat = scipy.io.loadmat(testlabels)
    
    names = mat['joint_names'][0]
    joints3D = mat['joint_xyz'][0]
    joints2D = mat['joint_uvd'][0]

    dmh = DepthMapHandle(588.03, 587.07, 320., 240., 300, 128)

    gtCropAll = np.zeros((joints3D.shape[0], numJoints*2), np.float32)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #for line in range(49165):
    i=0
    for line in range(joints3D.shape[0]):
        dptFileName = '{0:s}/depth_1_{1:07d}.png'.format(data_path, line+1)
        #dptFileName = '{0:s}/synthdepth_1_{1:07d}.png'.format(data_path, line+1)

        #dpt = dmh.loadDepthMap(dptFileName)
        dpt = dmh.cvLoadDepthMap(dptFileName)

        gtorig = np.zeros((numJoints, 3), np.float32)
        jt = 0
        for ii in range(joints2D.shape[1]):
            if ii not in jointIndex:
                continue
            gtorig[jt,0] = joints2D[line,ii,0]
            gtorig[jt,1] = joints2D[line,ii,1]
            gtorig[jt,2] = joints2D[line,ii,2]
            jt += 1
                    
        gt3Dorig = np.zeros((numJoints,3),np.float32)
        jt = 0
        for jj in range(joints3D.shape[1]):
            if jj not in jointIndex:
                continue
            gt3Dorig[jt,0] = joints3D[line,jj,0]
            gt3Dorig[jt,1] = joints3D[line,jj,1]
            gt3Dorig[jt,2] = joints3D[line,jj,2]
            jt += 1
      
        print line

        com = gtorig[13]
        #print line, com
        dpt = dmh.cropImage(dpt, com)
       
        dpt  = dmh.saveNormalizedDepthMap(dpt, com)

        saveDpt = np.zeros((dpt.shape[0],dpt.shape[1], 3), np.float32)
 
        saveDpt[:,:,0] = dpt;
        saveDpt[:,:,1] = dpt;
        saveDpt[:,:,2] = dpt;

        
        #Imread(saveDpt).convert('uint8')
        #print "saveDpt {}".format(saveDpt.shape)
        
        saveDpt_ = cv2.cvtColor(saveDpt, cv2.COLOR_BGR2GRAY)*255

        # for i in range(0,128):
        #     for j in range(0,128):
        #         if saveDpt_[i][j] < 0:
        #             saveDpt_[i][j] = 255



        saveName = "./data/hand/nyu_hand/png_cropped_dataset/normalized_real_data/crop_{0:05d}.png".format(line)

        #dpt = dpt * 255


        print saveDpt_
        cv2.imwrite(saveName, saveDpt_)

        #cv2.imwrite(saveName, dpt)
        #cv2.imshow("image", np.asarray(dpt, np.uint8))
        #cv2.imshow("image", dpt)
        #cv2.waitKey()

        i=i+1

    np.savetxt('joint_partial.txt', gtCropAll, fmt='%3.3f')