#!python

'''
Created on 31/03/2016

@author: diego
'''

from numpy import *
from numpy.random import *
import cv2


if __name__ == "__main__":
    
    nFiles = 2000
    
    for i in range(1,nFiles+1):
        if i < 10:
            filename = 'TrackingFrames/0000' + str(i) + '.png'
            filename_ = 'TrackingFrames/0000' + str(i) + '_.png'
            filename_ir = 'TrackingFrames/0000' + str(i) + '_ir.png'
            filename_ir_ = 'TrackingFrames/0000' + str(i) + '_ir_.png'
            filename_user = 'TrackingFrames/0000' + str(i) + '_user.png'
        elif i < 100:
            filename = 'TrackingFrames/000' + str(i) + '.png'
            filename_ = 'TrackingFrames/000' + str(i) + '_.png'
            filename_ir = 'TrackingFrames/000' + str(i) + '_ir.png'
            filename_ir_ = 'TrackingFrames/000' + str(i) + '_ir_.png'
            filename_user = 'TrackingFrames/000' + str(i) + '_user.png'
        elif i < 1000:
            filename = 'TrackingFrames/00' + str(i) + '.png'
            filename_ = 'TrackingFrames/00' + str(i) + '_.png'
            filename_ir = 'TrackingFrames/00' + str(i) + '_ir.png'
            filename_ir_ = 'TrackingFrames/00' + str(i) + '_ir_.png'
            filename_user = 'TrackingFrames/00' + str(i) + '_user.png'
        elif i < 10000:
            filename = 'TrackingFrames/0' + str(i) + '.png'
            filename_ = 'TrackingFrames/0' + str(i) + '_.png'
            filename_ir = 'TrackingFrames/0' + str(i) + '_ir.png'
            filename_ir_ = 'TrackingFrames/0' + str(i) + '_ir_.png'
            filename_user = 'TrackingFrames/0' + str(i) + '_user.png'
        else:        
            filename = 'TrackingFrames/' + str(i) + '.png'
            filename_ = 'TrackingFrames/' + str(i) + '_.png'
            filename_ir = 'TrackingFrames/' + str(i) + '_ir.png'
            filename_ir_ = 'TrackingFrames/' + str(i) + '_ir_.png'
            filename_user = 'TrackingFrames/' + str(i) + '_user.png'
            
        frame_depth = cv2.imread(filename,-1)
        frame_depth_edited = cv2.imread(filename_,-1)
        cv2.imshow('frame', frame_depth_edited)
        cv2.waitKey(10)

    
    