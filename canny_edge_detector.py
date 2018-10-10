# -*- coding: utf-8 -*-
# @Time    : 2018/10/2 22:34
# @Author  : Zhao Zhiwei
# @email   : 1596169007@qq.com
# @File    : canny_edge_detector.py

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import copy


class CannyEdgeDetector(object):

    """
    A naive canny edge detector's implementations:

    Canny Edge Detect Algorithms Procedure:
    Step1: Apply gauss filter to smooth image in order to denoise
    Step2: Generate gradient intensity and phase map for smoothed image
    Step3: Apply non-maximum suppression(NMS) to get rid of spurious response to edge detection(Slim edge)
    Step4: Apply double threshold to determine potentials edges: (strong and weak)
    step5: Track edge by hysteresis:
    Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

    see this paper for more details :
    "A Computational Approach to Edge Detection" ,IEEE Transactions on Pattern Analysis and Machine Intelligence ( Volume: PAMI-8 , Issue: 6 , Nov. 1986 )
    """

    def __init__(self,img_path,kernel_size,sigma,threshold):
        """
        :param img_path: /path/to/image
        :param kernel_size: gauss'kernel's size
        :param sigma: gauss kernel's variance
        :param threshold: [low_threshold, high_threshold] for step4 ,eg:[0.2,0.6]

        """
        self.kernel_size = kernel_size
        self.sigma = sigma

        # define a set of kernel
        self.kernel_dic = {
                            'sobel_x': np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
                            'sobel_y': np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),

                          }

        self.threshold = threshold

        # read image with gray mode
        self.img = cv2.imread(img_path,0)

        self.height = self.img.shape[0]
        self.width = self.img.shape[1]


    def gauss_filter(self):
        """
        Step1: Apply gauss filter to smooth image based on given gauss_kernel
        :return: gauss-blur img
        """
        img_blur = cv2.GaussianBlur(self.img,self.kernel_size,self.sigma)
        return img_blur

    def conv(self,img,filter_name):
        """
        implement a simple 2D-Conv operations
        :param filter_name: filter's name
        :return: feature map
        """
        mask = self.kernel_dic[filter_name]
        res = np.zeros((self.height,self.width))

        h_size = mask.shape[0]
        w_size = mask.shape[1]

        for h in range(self.height):
            for w in range(self.width):
                field = img[h:h+h_size, w:w+w_size]
                res[h][w] = np.sum(field*mask)

        return res


    def Gradients_Compute(self,img):
        """
        Step2: Generate gradient intensity and phase map for smoothed image

        sobel_x:   [-1 0 1;
                   -2 0 2;
                   -1 0 1];

        sobel_y:   [1 2 1;
                    0 0 0;
                  -1 -2 -1];
        :return:
        """

        h_size = 3
        w_size = 3
        # pad img to keep same size
        p = int((h_size-1)/2)
        q = int((w_size-1)/2)
        img_pad = np.pad(img,((p,p),(q,q)),'constant')

        grad_x = np.abs(self.conv(img_pad, 'sobel_x'))
        grad_y = np.abs(self.conv(img_pad, 'sobel_y'))

        grad_abs = np.sqrt(grad_x**2+grad_y**2)
        grad_phase = np.arctan2(grad_y,grad_x)*180/np.pi

        return grad_abs, grad_x, grad_y, grad_phase

    def NMS(self, det, phase):
        """
        apply NMS to every pixel
        :param det: grad_abs
        :param phase: grad_phase
        :return: pixel after NMS
        """
        gmax = np.zeros(det.shape)
        for i in range(gmax.shape[0]):
            for j in range(gmax.shape[1]):
                if phase[i][j] < 0:
                    phase[i][j] += 360
                # In order to beyond boarder
                if ((j + 1) < gmax.shape[1]) and ((j - 1) >= 0) and ((i + 1) < gmax.shape[0]) and ((i - 1) >= 0):
                    # 0 degrees
                    if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
                        # compare with respond two pixels
                        if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                            gmax[i][j] = det[i][j]
                    # 45 degrees
                    if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                        if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
                            gmax[i][j] = det[i][j]
                    # 90 degrees
                    if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                        if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
                            gmax[i][j] = det[i][j]
                    # 135 degrees
                    if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                        if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
                            gmax[i][j] = det[i][j]
        return gmax

    def double_threshold(self,grad_nms):
        """
        procedure:
        if grad_intensity[i][j]>high_threshold:
            grad_intensity[i][j] = high_threshold
        elif grad_intensity[i][j]>low_threshold:
            grad_intensity[i][j] = low_threshold
        else:
            grad_intensity[i][j] = 0

        :param grad_nms: grad_intensity after nms
        :return: a grad_intensity which contains [weak,strong]
                 a flag array record edge's locations

        """
        # a flag
        is_edge = np.zeros(grad_nms.shape)
        grad_th = np.zeros(grad_nms.shape)
        strong = 1.0
        weak = 0.4
        mmax = np.max(grad_nms)
        print(grad_nms)

        low_threshold = self.threshold[0]*mmax
        high_threshold = self.threshold[1]*mmax

        for i in range(grad_nms.shape[0]):
            for j in range(grad_nms.shape[1]):
                if grad_nms[i][j]>=high_threshold:
                    grad_th[i][j] = strong
                    is_edge[i][j] = 1.0
                elif grad_nms[i][j] >= low_threshold:
                    grad_th[i][j] = weak
                else :
                    grad_th[i][j] = 0

        return grad_th, is_edge

    def track(self, gradmCopy, is_edge):
        """

        :param gradmCopy: grad_intensity after double threshold
        :param is_edge: A flag matrix record edge's locations
        :return: A flag matrix record edge's locations
        """
        strong = 1.0
        weak = 0.4
        height = gradmCopy.shape[0]
        width = gradmCopy.shape[1]

        for ii in range(1,height-2):
            for jj in range(1,width-2):

                flag = True

                if gradmCopy[ii][jj] == weak:

                    neighbors = [
                                 [gradmCopy[ii - 1][jj - 1], gradmCopy[ii - 1][jj], gradmCopy[ii - 1][jj + 1] ],
                                 [gradmCopy[ii][jj - 1], 0, gradmCopy[ii][jj + 1] ],
                                 [gradmCopy[ii + 1][jj - 1], gradmCopy[ii + 1][jj], gradmCopy[ii + 1][jj + 1]]
                                 ]

                    for i in range(3):
                        for j in range(3):
                            if neighbors[i][j] == strong:
                                flag = True
                                break

                    if flag==True:
                        is_edge[ii][jj] = 1.0

        return is_edge


img = cv2.imread('./data/fly.jpg',0)
img_gt = cv2.imread('./data/flygt.jpg',0)

ced = CannyEdgeDetector('./data/fly.jpg',(5,5), 1.4, (0.10,0.60))

blur = cv2.GaussianBlur(img,(5,5),1.4)
canny = cv2.Canny(img,130,240)


img_blur = ced.gauss_filter()
grad_abs, grad_x, grad_y, grad_phase = ced.Gradients_Compute(img_blur)
grad_nms = ced.NMS(grad_abs,grad_phase)
grad_th, is_edge = ced.double_threshold(grad_nms)
edge = ced.track(grad_th,is_edge)


plt.figure(figsize=(10,5))
plt.subplot(221)
plt.imshow(img_blur,cmap='gray')
plt.title('Gauss blur')
plt.subplot(223)
plt.imshow(grad_x,cmap='gray')
plt.title('vertical Edge')
plt.subplot(224)
plt.imshow(grad_y,cmap='gray')
plt.title('horizontal Edge')
plt.subplot(222)
plt.imshow(grad_abs,cmap='gray')
plt.title('Edge')


plt.figure(figsize=(10,5))

plt.subplot(231)
plt.imshow(img,cmap='gray')
plt.title('Origin')

plt.subplot(232)
plt.imshow(img_blur,cmap='gray')
plt.title('Step1:Gauss Blur')

plt.subplot(233)
plt.imshow(grad_abs,cmap='gray')
plt.title('Step2:Gradients')

plt.subplot(234)
plt.imshow(grad_nms,cmap='gray')
plt.title('Step3:NMS')

plt.subplot(235)
plt.imshow(grad_th,cmap='gray')
plt.title('Step4:Double-Threshold')

plt.subplot(236)
plt.imshow(edge,cmap='gray')
print(edge)
plt.title('Step5:Track-Final')

plt.figure(figsize=(12,7))

plt.subplot(221)
plt.imshow(img,cmap='gray')
plt.title('Original')

plt.subplot(222)
plt.imshow(img_gt,cmap='gray')
plt.title('Ground Truth')

plt.subplot(223)
plt.imshow(canny,cmap='gray')
plt.title('Opencv Result')

plt.subplot(224)
plt.imshow(edge,cmap='gray')
plt.title('My Result')

plt.show()

































