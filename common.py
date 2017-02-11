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

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def show_images(im1, im2, title1, title2, window_title, cmap1=None, cmap2=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, squeeze=True)
    fig.canvas.set_window_title(window_title)
    if cmap1 is None:
        ax1.imshow(im1)
    else:
        ax1.imshow(im1, cmap=cmap1)
    if cmap2 is None:
        ax2.imshow(im2)
    else:
        ax2.imshow(im2, cmap=cmap2)
    ax1.set_title(title1)
    ax2.set_title(title2)
    fig.tight_layout()
    fig.show()

def equalize_hist(im, show=False):
    im_eq = cv.cvtColor(im, cv.COLOR_RGB2YCrCb)
    zeros = np.zeros(im_eq.shape[:2])
    im_eq[:,:,0] = cv.equalizeHist(im_eq[:,:,0], zeros)
    im_eq = cv.cvtColor(im_eq, cv.COLOR_YCrCb2RGB)
    if show == True:
        show_images(im, im_eq, 'original', 'equalized', 'Histogram equalization')
    return im_eq

