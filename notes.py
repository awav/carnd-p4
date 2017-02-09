# The MIT License (MIT)
# 
# Copyright 2017 Artem Artemev, im@artemav.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from calibrator import Calibrator as clb
from common import show_images

class Perspective():
    _src, _dst = [], []
    _perspective = None
    _perspective_inv = None
    def __new__(cls, *args, **kvargs):
        raise ValueError("You can't create `Calibration` instance")
    @classmethod
    def set_src_points(cls, src):
        cls._src = src
    @classmethod
    def set_dst_points(cls, dst):
        cls._dst = dst
    @classmethod
    def find_perspective(cls, src=None, dst=None):
        cls._src = cls._src if src == None else src
        cls._dst = cls._dst if dst == None else dst
        cls._perspective = cv.getPerspectiveTransform(cls._src, cls._dst)
        cls._perspective_inv = cv.getPerspectiveTransform(cls._dst, cls._src)
    @classmethod
    def warp(im, inverse=False, show=False):
        assert(cls._perspective is not None)
        assert(cls._perspective is not None)
        imsize = im.shape[1], im.shape[0]
        if inverse == False:
            warped = cv.warpPerspective(im, cls._perspective,
                imsize, flags=cv.INTER_LINEAR)
        else:
            warped = cv.warpPerspective(im, cls._perspective_inv,
                imsize, flags=cv.INTER_LINEAR)
        if show == True:
            pts = np.int32(cls._src).reshape((-1,1,2))
            impoly = cv.polylines(im, [pts], True, (0,255,0))
            show_images(impoly, warped, 'original', 'warped', 'perspective transform')
        return warped

def redchannel(im, show=False):
    R = im[:,:,0]
    if show == True:
        show_images(im, R, 'original', 'red channel', 'RGB', cmap2='gray')
    return R

def saturation(im, show=False):
    S = cv.cvtColor(im, cv.COLOR_RGB2HLS)[:,:,2]
    if show == True:
        show_images(im, S, 'original', 'saturation', 'HLS', cmap2='gray')
    return S

def corners_unwarp(im, src, dst):
    undist_im = clb.undistort(im)
    gray = cv.cvtColor(undist_im, cv.COLOR_RGB2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        cv.drawChessboardCorners(undist_im, (nx, ny), corners, ret)
        plt.imshow(undist_im)
        ox, oy = [0] * 2
        dox, doy, dex, dey = 100, 100, 1200, 900  
        src_ids = [[nx-1,oy],[nx-1,ny-1],[ox,ny-1],[ox,oy]]
        src_ids = flat_indices(src_ids, (nx, ny))
        src = corners[src_ids]
        dst = np.float32([[dex,doy],[dex,dey],[dox,dey],[dox, doy]])
        M = cv.getPerspectiveTransform(src, dst)
        im_size = im.shape[1], im.shape[0]
        warped = cv.warpPerspective(undist_im, M, im_size, flags=cv.INTER_LINEAR)
  
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20, 100), show=False):
    #gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    x, y = (1, 0) if orient == "x" else (0, 1)
    sobel = cv.Sobel(gray, cv.CV_64F, x, y, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    thresh_min, thresh_max = thresh
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    if show == True:
        show_images(gray, binary, 'origin', 'sobel', 'sobel threshold', cmap1='gray', cmap2='gray')
    return binary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(30, 100), show=False):
    #gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
    sobely = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    thresh_min, thresh_max = mag_thresh
    binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    if show == True:
        show_images(gray, binary, 'origin', 'mag', 'mag threshold', cmap1='gray', cmap2='gray')
    return binary

def dir_threshold(gray, sobel_kernel=3, thresh=(.7, 1.3), show=False):
    #gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    sobx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobx = np.absolute(sobx)
    abs_soby = np.absolute(soby)
    arctan = np.arctan2(abs_soby, abs_sobx)
    thr_min, thr_max = thresh
    binary = np.zeros_like(arctan)
    binary[(arctan >= thr_min) & (arctan <= thr_max)] = 1
    if show == True:
        show_images(gray, binary, 'origin', 'dir', 'dir threshold', cmap1='gray', cmap2='gray')
    return binary

def thresholds(im):
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]

    thresh_gray = (180, 255)
    binary_gray = np.zeros_like(gray)
    binary_gray[(gray > thresh_gray[0]) & (gray <= thresh_gray[1])] = 1

    h = (200, 255)
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
     
    hls = cv.cvtColor(im, cv.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    thresh_s = (90, 255) # (170, 255)
    binary_s = np.zeros_like(S)
    binary_s[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1
    
    thresh_h = (15, 100)
    binary_h = np.zeros_like(H)
    binary_h[(H > thresh_h[0]) & (H <= thresh_h[1])] = 1

# Edit this function to create your own pipeline.
def pipeline(im, s_thresh=(170, 255), sx_thresh=(20, 100)):
    im = np.copy(im)
    # Convert to HSV color space and separate the V channel
    hsv = cv.cvtColor(im, cv.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

# notes.run(im, ksize=5, tdir=(0.8,1.2), ty=(20,100), tx=(20,100), tmag=(40,100)) ???
def run(im, ksize=3, tx=(20, 100), ty=(20, 100), tmag=(30, 100), tdir=(0.9, 1.1), show=False):
    ksize=3
    gradx = abs_sobel_thresh(im, orient='x', sobel_kernel=ksize, thresh=tx)
    grady = abs_sobel_thresh(im, orient='y', sobel_kernel=ksize, thresh=ty)
    mag_binary = mag_thresh(im, sobel_kernel=ksize, mag_thresh=tmag)
    dir_binary = dir_threshold(im, sobel_kernel=ksize, thresh=tdir)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    if show == True:
        show_images(im, binary, 'origin', 'combined', 'combination', cmap1='gray', cmap2='gray')


#
#Steps we’ve covered so far:
#
#Camera calibration
#Distortion correction
#Color/gradient threshold
#Perspective transform
#After doing these steps, you’ll be given two additional steps for the project:
#
#Detect lane lines
#Determine the lane curvature

