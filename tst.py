#!/usr/bin/env python

import theano
from theano import tensor as T
#import matplotlib as plt
import numpy as np
import random
import pylab
import lasagne

import scipy
from contextlib import contextmanager
import os, glob

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def loadImages():
    baseDir = '/l/SCRATCH/hashai28/theano_ou/kaggle_eye/sample/'
    baseDir = './kaggle_eye/sample/'
    with cd(baseDir):
        fList = glob.glob('*.jpeg')
        im = []
        for fn in fList:
            curIm = pylab.imread(fn)
            curIm = scipy.misc.imresize(curIm, (600,600))
            im.append(curIm)
        im = np.asarray(im, dtype='float32').transpose(0,3,1,2)
    return im

def mk1():
   images = T.ftensor4('images')
   z = T.nnet.neighbours.images2neibs(images, neib_shape=(3,3))
   #, mode='ignore_borders' )
   im2Neibs = theano.function([images],[z])
   return im2Neibs

def mk2(shape):
   neibs = T.fmatrix('neibs')
   z = T.nnet.neighbours.neibs2images(neibs, (3,3), shape)
   #, mode='wrap_centered')
   neibs2Im = theano.function([neibs],[z])
   return neibs2Im
   
 def lpb1(neibs):
     centerInd = neibs.shape[1] / 2
     centerVals = neibs[ : , centerInd ]
     t = neib - centerVals.reshape((len(centerVals)),1)
     return np.log( (t + abs(t))/2.  + 1. ) 


def main(): 
   im = loadImages()
   im2Neibs = mk1()  
   neibs2Im = mk2(im.shape)  
   neib = im2Neibs(im)[0]
   l = lpb1(neib)
   w = neibs2Im(neib)[0]

    
    

if __name__=="__main__":
    main()

