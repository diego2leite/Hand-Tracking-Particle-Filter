#!python

'''
Created on 31/03/2016

@author: diego
'''

import cv2
import numpy as np
import numpy.random as rnd
import itertools


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = rnd.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices

            
def particlefilter(sequence, pos, stepsize, n):
    seq = iter(sequence)
    x = np.ones((n, 2), int) * pos                   # Initial position
    f0 = seq.next()[tuple(pos)] * np.ones(n)         # Target colour model
    yield pos, x, np.ones(n)/n                       # Return expected position, particles and weights
    for im in seq:
        x += rnd.uniform(-stepsize, stepsize, x.shape)  # Particle motion model: uniform step
        x  = x.clip(np.zeros(2), np.array(im.shape)-1).astype(int) # Clip out-of-bounds particles
        f  = im[tuple(x.T)]                         # Measure particle colours
        w  = 1./(1. + (f0-f)**2)                    # Weight~ inverse quadratic colour distance
        w /= sum(w)                                 # Normalize w
        yield sum(x.T*w, axis=1), x, w              # Return expected position, particles and weights
        if 1./sum(w**2) < n/2.:                     # If particle cloud degenerate:
            x  = x[resample(w),:]                     # Resample particles according to weights


if __name__ == "__main__":
    
    from pylab import *

    initialFrame = 95
    nFiles = 2000
     
    frames = []
     
    for i in range(initialFrame,nFiles+1):
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
         
#         if(i == 95):
#             frame = cv2.cvtColor(frame_depth_edited,cv2.COLOR_GRAY2RGB)
#              
#             cv2.circle(frame,(225,125), 5, (0,255,0), -1)
#             cv2.imshow('frame', frame)
#             cv2.waitKey()
         
        frames.append(frame_depth_edited)
         
        #cv2.imshow('frame', frame_depth_edited)
        #cv2.waitKey(10)
    
    ion()
    
    #seq = [ im for im in np.zeros((20,240,320), int)]      # Create an image sequence of 20 frames long
    seq = [ im for im in frames]      # Create an image sequence of 20 frames long
    
    x0 = np.array([125, 225])                              # Add a square with starting position x0 moving along trajectory xs
#     xs = np.vstack((np.arange(20)*3, np.arange(20)*2)).T + x0
#     for t, x in enumerate(xs):
#         xslice = slice(x[0]-8, x[0]+8)
#         yslice = slice(x[1]-8, x[1]+8)
#         seq[t][xslice, yslice] = 255
    
    i = 0
    
    for im, p in itertools.izip(seq, particlefilter(seq, x0, 25, 100)): # Track the square through the sequence
        print i
        i = i + 1
        
        pos, xs, ws = p
        position_overlay = np.zeros_like(im)
        position_overlay[tuple(pos)] = 1
        particle_overlay = np.zeros_like(im)
        particle_overlay[tuple(xs.T)] = 1
        
        hold(True)
        draw()
        #time.sleep(0.3)
        clf()                                           # Causes flickering, but without the spy plots aren't overwritten
        imshow(im,cmap=cm.gray)                         # Plot the image
        
        #spy(particle_overlay, marker='.', color='r')    # Plot the particles
        spy(position_overlay, marker='.', color='g')    # Plot the expected position
    show()
    
#     nFiles = 2000 - 1500
#     
#     seq = []
#     
#     for i in range(1,nFiles+1):
#         if i < 10:
#             filename = 'TrackingFrames/0000' + str(i) + '.png'
#             filename_ = 'TrackingFrames/0000' + str(i) + '_.png'
#             filename_ir = 'TrackingFrames/0000' + str(i) + '_ir.png'
#             filename_ir_ = 'TrackingFrames/0000' + str(i) + '_ir_.png'
#             filename_user = 'TrackingFrames/0000' + str(i) + '_user.png'
#         elif i < 100:
#             filename = 'TrackingFrames/000' + str(i) + '.png'
#             filename_ = 'TrackingFrames/000' + str(i) + '_.png'
#             filename_ir = 'TrackingFrames/000' + str(i) + '_ir.png'
#             filename_ir_ = 'TrackingFrames/000' + str(i) + '_ir_.png'
#             filename_user = 'TrackingFrames/000' + str(i) + '_user.png'
#         elif i < 1000:
#             filename = 'TrackingFrames/00' + str(i) + '.png'
#             filename_ = 'TrackingFrames/00' + str(i) + '_.png'
#             filename_ir = 'TrackingFrames/00' + str(i) + '_ir.png'
#             filename_ir_ = 'TrackingFrames/00' + str(i) + '_ir_.png'
#             filename_user = 'TrackingFrames/00' + str(i) + '_user.png'
#         elif i < 10000:
#             filename = 'TrackingFrames/0' + str(i) + '.png'
#             filename_ = 'TrackingFrames/0' + str(i) + '_.png'
#             filename_ir = 'TrackingFrames/0' + str(i) + '_ir.png'
#             filename_ir_ = 'TrackingFrames/0' + str(i) + '_ir_.png'
#             filename_user = 'TrackingFrames/0' + str(i) + '_user.png'
#         else:
#             filename = 'TrackingFrames/' + str(i) + '.png'
#             filename_ = 'TrackingFrames/' + str(i) + '_.png'
#             filename_ir = 'TrackingFrames/' + str(i) + '_ir.png'
#             filename_ir_ = 'TrackingFrames/' + str(i) + '_ir_.png'
#             filename_user = 'TrackingFrames/' + str(i) + '_user.png'
#             
#         frame_depth = cv2.imread(filename,-1)
#         frame_depth_edited = cv2.imread(filename_,-1)
#         
# #         if(i == 95):
# #             frame = cv2.cvtColor(frame_depth_edited,cv2.COLOR_GRAY2RGB)
# #              
# #             cv2.circle(frame,(225,125), 5, (0,255,0), -1)
# #             cv2.imshow('frame', frame)
# #             cv2.waitKey()
#         
#         seq.append(frame_depth_edited)
#         
#         #cv2.imshow('frame', frame_depth_edited)
#         #cv2.waitKey(10)
#     
#     
#     seq = [ im for im in np.zeros((20,240,320), int)]      # Create an image sequence of 20 frames long
#     
#     x0 = np.array([120, 160])                              # Add a square with starting position x0 moving along trajectory xs
#     xs = np.vstack((np.arange(20)*3, np.arange(20)*2)).T + x0
#     for t, x in enumerate(xs):
#         xslice = slice(x[0]-8, x[0]+8)
#         yslice = slice(x[1]-8, x[1]+8)
#         seq[t][xslice, yslice] = 255
#         
#         
#     #x0 = np.array([220, 120])   
# 
#     for im, p in itertools.izip(seq, particlefilter(seq, x0, 8, 100)): # Track the square through the sequence
#         #print p
#         pos, xs, ws = p
#         position_overlay = np.zeros_like(im)
#         position_overlay[tuple(pos)] = 1
#         particle_overlay = np.zeros_like(im)
#         particle_overlay[tuple(xs.T)] = 1
#         #hold(True)
#         #draw()
#         #time.sleep(0.3)
#         #clf()                                           # Causes flickering, but without the spy plots aren't overwritten
#         #imshow(im,cmap=cm.gray)                         # Plot the image
#         
#         #spy(particle_overlay, marker='.', color='r')    # Plot the particles
#         #spy(position_overlay, marker='.', color='b')    # Plot the expected position
#         
#         cv2.imshow('frame', im)
#         cv2.waitKey(10)
#         
#         
#     #show()
        
        

    
    