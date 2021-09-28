
import numpy as np
import cv2
import skimage
import sys
import os
from skimage import color
from skimage import measure

def qtgetblk(A,S,dim):
    Sind = np.array(np.where(S == dim))
    numBlocks = len(Sind[0])
    if numBlocks != 0:
        val=np.zeros((dim,dim,numBlocks))
        for i in range(numBlocks):
            val[:,:,i] = A[Sind[1][i]:Sind[1][i]+dim,Sind[0][i]:Sind[0][i]+dim]
    return val

def qtsetblk(A,S,dim,val):
    Sind = np.array(np.where(S == dim))
    numBlocks = len(Sind[0])
    if numBlocks != 0:
        for i in range(numBlocks):
            A[Sind[1][i]:Sind[1][i]+dim,Sind[0][i]:Sind[0][i]+dim] = val[:,:,i]
    return A

def QuadReconstructRefined(S, img, minDim):
    img = np.double(img)
    M = np.array([[-1,3,-3,1],
                [3,-6,3,0],
                [-3,3,0,0],
                [1,0,0,0]])
    newS = S + (S > 0)
    maxDim = np.max(newS)
    dim = maxDim
    newS = np.lib.pad(newS, 1)
    newImg = np.lib.pad(img, 1)
    tempReconstImg = np.zeros_like(newImg)
    while dim >= minDim + 1:
        newInd = np.where(newS == dim)
        length = len(newInd[0])
        if length != 0:
            # Get block with size Dim
            blks = qtgetblk(newImg, newS, dim)
            subDim = (dim - 1) / 4
            row = [0, subDim * 2-1, subDim * 3-1, subDim * 4]
            xx = np.array([row,row,row,row])
            yy = xx.T
            inds = sub2ind([dim, dim], yy, xx)
            inds = np.array(inds,np.uint32)
            U = np.array([[np.power(x/(dim-1), 3), np.power(x/(dim-1), 2), x/(dim-1), np.power(x/(dim-1), 0)] for x in range(dim)])
            # Left Matrix: UM
            # Right Matrix: MU`
            LM = np.matmul(U,M)
            RM = np.matmul(M.T,U.T)
            reBlkSeq = np.zeros((dim, dim, length))
            for ii in range(length):
                blockVal = blks[:,:,ii].flatten()
                blockVal = blockVal[inds.T]
                reblkVal = np.matmul(np.matmul(LM,blockVal),RM)
                reBlkSeq[:,:,ii] = reblkVal
            tempReconstImg = qtsetblk(tempReconstImg,newS,dim,reBlkSeq)
        dim = int((dim - 1) / 2 + 1)
    bgImg = tempReconstImg[1:-1, 1:-1]
    return bgImg

def sub2ind(array_shape, rows, cols):
    ind = rows + cols*array_shape[0]
    return ind

def splitImage(inImg):
    h,w = inImg.shape[0], inImg.shape[1]
    off1X=0
    off1Y=0
    off2X=int(w/2)
    off2Y=int(h/2)
    
    img1 = inImg[0:off2Y, 0:off2X]
    img2 = inImg[0:off2Y, off2X:w]
    img3 = inImg[off2Y:h, 0:off2X]
    img4 = inImg[off2Y:h, off2X:w]
    
    return off1X,off1Y,off2X,off2Y,img1,img2,img3,img4

class theQTree:
    def qt(self, inImg, minStd, minSize, offX, offY):
        h,w = inImg.shape[0], inImg.shape[1]  
        m,s = cv2.meanStdDev(inImg)
        temp = np.max(inImg) - np.min(inImg)
        if temp>minStd*255 and max(h,w)>minSize:
            oX1,oY1,oX2,oY2,im1,im2,im3,im4 = splitImage(inImg)
            
            self.qt(im1, minStd, minSize, offX+oX1, offY+oY1)
            self.qt(im2, minStd, minSize, offX+oX2, offY+oY1)
            self.qt(im3, minStd, minSize, offX+oX1, offY+oY2)
            self.qt(im4, minStd, minSize, offX+oX2, offY+oY2)
        else:
            self.listRoi.append([offX,offY,w,h,m,s])

    def __init__(self, img, stdmin, sizemin):
        self.listRoi = []
        self.qt(img, stdmin, sizemin, 0, 0)

def entropy(image):
    image = skimage.img_as_float(image)
    grayscale = skimage.color.rgb2gray(image)
    entropy1 = skimage.measure.shannon_entropy(grayscale)
    return entropy1

def main():
    # Parameters
    minDev   = 0.2
    minSz    = 32

    Ori_path = './asserts/testdata/'
    sal_path = './asserts/saliencyMaps/'
    save_path = './asserts/fusionRresults/'
    if not os.path.exists(save_path): os.mkdir(save_path)
    imgs = os.listdir(sal_path)
    for img_name in imgs:
        img_name = img_name.split('.')[0]
        img_name = img_name.split('_')[0]
        img_name = img_name+'.jpg'
        OriVis= cv2.imread(Ori_path+'RGB/'+img_name)
        OriIR = cv2.imread(Ori_path+'T/'+img_name)
        salVis = cv2.imread(sal_path+img_name.split('.')[0]+'_1.png')
        salIR = cv2.imread(sal_path+img_name.split('.')[0]+'_2.png')

        import time
        st = time.time()
        # input values
        h,w = salIR.shape[0], salIR.shape[1]
        salIR = cv2.GaussianBlur(salIR,(5,5),3)
        
        I = cv2.resize(salIR,(512,512))
        
        if not type(I)==type(None):
            if I.ndim > 1 :
                img = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            else :
                img = I
            
        print('execution...'+img_name)
        # QuadTree Decomposition
        qt=theQTree(img,minDev,minSz)
        rois = qt.listRoi

        S = np.zeros_like(img)

        Qimg = img.copy()
        
        for e in rois:
            S[e[0],e[1]] = e[2]
            cv2.rectangle(Qimg, (e[0],e[1]), (e[0]+e[2],e[1]+e[3]), 125, 1)

        cv2.imwrite(save_path+img_name[:-4]+'_Quad.png',np.uint8(Qimg))
        kernel = np.ones((int(minSz), int(minSz)),np.uint8)
        minImg = cv2.erode(img, kernel)

        # Bezier Reconstruction
        bgImg = QuadReconstructRefined(S, minImg, minSz)
        bgImg = np.array(bgImg,dtype='uint8')
        bgImg = cv2.resize(bgImg, (w,h))

        bgImg = cv2.GaussianBlur(bgImg,(3,3),3/2)
        # Adaptive fusion
        salVis = cv2.cvtColor(salVis, cv2.COLOR_BGR2GRAY)
        addFeature = np.double(bgImg) * (entropy(OriIR) / entropy(OriVis))
        addFeaturelogical = addFeature > 0
        addedVals = np.double(addFeature) * addFeaturelogical + np.double(salVis)
        maxVals = np.sort(addedVals.flatten())[::-1]
        maxMean = np.mean(maxVals[0 : round(0.001 * len(maxVals))])
        ratio = min(255 / maxMean, 0.4)
        result = np.double(np.double(salVis) + np.double(addFeature * ratio))
        result[result>255] = 255
        et = time.time()
        print(et-st)
        cv2.imwrite(save_path+img_name[:-4]+'.png',np.uint8(result))

if __name__ == '__main__':
    main()
