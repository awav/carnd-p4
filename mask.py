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
from perspective import Perspective as persp
from common import show_images

class CustomMask():
    @staticmethod
    def apply(im, show=False):
        s = ColorThresholder.hls_channel(im, channel='s', thresh=(100, 255))
        l = ColorThresholder.hls_channel(im, channel='l', thresh=(200, 255))
        g = ColorThresholder.rgb_channel(im, channel='g', thresh=(200, 255))
        m = SobelMask.magnitude(im, thresh=(15,255))
        #m = np.zeros(s.shape)
        since = im.shape[0] - (im.shape[0]//2)
        m[since:,:] = 0
        #l = np.zeros(s.shape)
        binary = np.zeros(s.shape)
        binary[(s > 0) | (l > 0) | (g > 0) | (m > 0)] = 1
        if show == True:
            combined = (g == m).astype(np.uint8)
            binary_color = np.dstack([s, l, combined])
            binary_color[binary_color == 1] = 100
            show_images(im, binary_color, 'original', 'binary colored', 'Masking image')
            show_images(im, binary, 'original', 'binary b&w', 'Masking image')
        return binary

class ColorThresholder():
    @staticmethod
    def rgb_channel(im, channel='s', thresh=(0,255), show=False):
        i, title = 0, 'red'
        if channel == 'g':
            i, title = 1, 'green'
        elif channel == 'b':
            i, title = 2, 'blue'
        chan = np.copy(im[:,:,i])
        chan[(chan < thresh[0]) | (chan > thresh[1])] = 0
        if show == True:
            title = '{0}, thresh={1}'.format(title, thresh)
            show_images(im, chan, 'original', title, 'RGB', cmap2='gray')
        return chan
    @staticmethod
    def hls_channel(im, channel='s', thresh=(0,255), show=False):
        i, title = 2, 'saturation'
        if channel == 'h':
            i, title = 0, 'hue'
        elif channel == 'l':
            i, title = 1, 'level'
        chan = cv.cvtColor(im, cv.COLOR_RGB2HLS)[:,:,i]
        chan[(chan < thresh[0]) | (chan > thresh[1])] = 0
        if show == True:
            title = '{0}, thresh={1}'.format(title, thresh)
            show_images(im, chan, 'original', title, 'HLS', cmap2='gray')
        return chan


class SobelMask():
    @staticmethod
    def abs(im, orient='x', sobel_kernel=3, thresh=(20, 100), show=False):
        gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        x, y = (1, 0) if orient == "x" else (0, 1)
        sobel = cv.Sobel(gray, cv.CV_64F, x, y, ksize=sobel_kernel)
        abs = np.absolute(sobel)
        scaled = np.uint8(255 * abs / np.max(abs))
        tmin, tmax = thresh
        binary = np.zeros_like(scaled)
        binary[(scaled >= tmin) & (scaled <= tmax)] = 1
        if show == True:
            show_images(gray, binary, 'origin', 'sobel',
                        'sobel thresh',
                        cmap1='gray', cmap2='gray')
        return binary
    @staticmethod
    def magnitude(im, sobel_kernel=3, thresh=(30, 100), show=False):
        gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        sobelx = cv.Sobel(gray, cv.CV_64F, dx=1, dy=0, ksize=sobel_kernel)
        sobely = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1, ksize=sobel_kernel)
        abs = np.sqrt(sobelx**2 + sobely**2)
        scaled = np.uint8(255 * abs / np.max(abs))
        binary = np.zeros_like(scaled)
        tmin, tmax = thresh
        binary[(scaled >= tmin) & (scaled <= tmax)] = 1
        if show == True:
            show_images(gray, binary, 'origin', 'magnitude',
                        'Magnitude Thresholding',
                        cmap1='gray', cmap2='gray')
        return binary
    @staticmethod
    def direction(im, sobel_kernel=3, thresh=(.7, 1.3), show=False):
        gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        sobx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
        soby = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobx = np.absolute(sobx)
        abs_soby = np.absolute(soby)
        arctan = np.arctan2(abs_soby, abs_sobx)
        tmin, tmax = thresh
        binary = np.zeros_like(arctan)
        binary[(arctan >= tmin) & (arctan <= tmax)] = 1
        if show == True:
            show_images(gray, binary, 'origin', 'direction',
                        'Direction sobel',
                        cmap1='gray', cmap2='gray')
        return binary

##def threshs(im):
##    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
##    R = image[:,:,0]
##    G = image[:,:,1]
##    B = image[:,:,2]
##
##    thresh_gray = (180, 255)
##    binary_gray = np.zeros_like(gray)
##    binary_gray[(gray > thresh_gray[0]) & (gray <= thresh_gray[1])] = 1
##
##    h = (200, 255)
##    binary = np.zeros_like(R)
##    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
##
##    hls = cv.cvtColor(im, cv.COLOR_RGB2HLS)
##    H = hls[:,:,0]
##    L = hls[:,:,1]
##    S = hls[:,:,2]
##
##    thresh_s = (90, 255) # (170, 255)
##    binary_s = np.zeros_like(S)
##    binary_s[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1
##
##    thresh_h = (15, 100)
##    binary_h = np.zeros_like(H)
##    binary_h[(H > thresh_h[0]) & (H <= thresh_h[1])] = 1
##
### Edit this function to create your own pipeline.
##def pipeline(im, s_thresh=(170, 255), sx_thresh=(20, 100)):
##    im = np.copy(im)
##    # Convert to HSV color space and separate the V channel
##    hsv = cv.cvtColor(im, cv.COLOR_RGB2HLS).astype(np.float)
##    l_channel = hsv[:,:,1]
##    s_channel = hsv[:,:,2]
##    # Sobel x
##    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 0) # Take the derivative in x
##    absx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
##    scaled = np.uint8(255*absx/np.max(absx))
##
##    # Threshold x gradient
##    sxbinary = np.zeros_like(scaled)
##    sxbinary[(scaled >= sx_thresh[0]) & (scaled <= sx_thresh[1])] = 1
##
##    # Threshold color channel
##    s_binary = np.zeros_like(s_channel)
##    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
##    # Stack each channel
##    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
##    # be beneficial to replace this channel with something else.
##    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
##    return color_binary
##
### notes.run(im, ksize=5, tdir=(0.8,1.2), ty=(20,100), tx=(20,100), tmag=(40,100)) ???
##def run(im, ksize=3, tx=(20, 100), ty=(20, 100), tmag=(30, 100), tdir=(0.9, 1.1), show=False):
##    ksize=3
##    gradx = abs_thresh(im, orient='x', sobel_kernel=ksize, thresh=tx)
##    grady = abs_thresh(im, orient='y', sobel_kernel=ksize, thresh=ty)
##    mag_binary = mag_thresh(im, sobel_kernel=ksize, mag_thresh=tmag)
##    dir_binary = dir_thresh(im, sobel_kernel=ksize, thresh=tdir)
##    combined = np.zeros_like(dir_binary)
##    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
##    if show == True:
##        show_images(im, binary, 'origin', 'combined', 'combination', cmap1='gray', cmap2='gray')
