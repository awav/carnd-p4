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
import mask

def nonzero_points(nonz, left=((0,200),(720,600)), right=((0,800), (720,1200))):
    # lty => left top y
    # ltx => left top x
    # lby => left bottom y
    # lbx => left bottom x
    (lty, ltx), (lby, lbx) = left
    (rty, rtx), (rby, rbx) = right
    l = (nonz[0] >= lty) & (nonz[0] <= lby) & (nonz[1] >= ltx) & (nonz[1] <= lbx)
    r = (nonz[0] >= rty) & (nonz[0] <= rby) & (nonz[1] >= rtx) & (nonz[1] <= rbx)
    nonz_l = nonz[:, l]
    nonz_r = nonz[:, r]
    return (nonz_l[0], nonz_l[1]), (nonz_r[0], nonz_r[1])

def fit_lanes(im, show=False):
    nonz = np.array(im.nonzero())
    (pyl, pxl), (pyr, pxr) = nonzero_points(nonz)
    fit_l = np.polyfit(pyl, pxl, 2)
    fit_r = np.polyfit(pyr, pxr, 2)
    if show == True:
        y_axe = np.linspace(0, im.shape[0] - 1, im.shape[0])
        line_l = fit_l[0] * y_axe**2 + fit_l[1] * y_axe + fit_l[2]
        line_r = fit_r[0] * y_axe**2 + fit_r[1] * y_axe + fit_r[2]
        ps_l = np.array([np.transpose(np.vstack([line_l, y_axe]))])
        ps_r = np.array([np.flipud(np.transpose(np.vstack([line_r, y_axe])))])
        points = np.hstack([ps_l, ps_r])
        draw = np.zeros_like(im).astype(np.uint8)
        cv.fillPoly(draw, np.int32([points]), (255,255,0))
        output = cv.addWeighted(np.uint8(im), 1, draw, 0.1, 0)
        show_images(im, output, 'origin', 'lanes', 'Fit lines')
    return fit_l, fit_r

def test(filename, n = 0, show=False):
    import importlib
    importlib.reload(mask)
    print(persp._perspective)
    im_ = plt.imread(filename);
    im = clb.undistort(im_);
    if n == -1:
       return persp.warp(im, show=show)
    elif n == 0:
       im2 = persp.warp(im, show=show)
       mask.hls_channel(im2, channel='h', threshold=(100, 255), show=show)
       mask.hls_channel(im2, channel='l', threshold=(200, 255), show=show)
       mask.hls_channel(im2, channel='s', threshold=(100, 255), show=show)
    elif n == 1:
       im_eq = equalize_hist(im, show=show)
       im2 = persp.warp(im_eq, show=show)
       mask.hls_channel(im2, channel='h', threshold=(0, 100), show=show)
       mask.hls_channel(im2, channel='l', threshold=(200, 255), show=show)
       mask.hls_channel(im2, channel='s', threshold=(100, 255), show=show)
    elif n == 2:
       im2 = persp.warp(im, show=show)
       mask.rgb_channel(im2, channel='r', threshold=(220, 255), show=show)
       mask.rgb_channel(im2, channel='g', threshold=(210, 255), show=show)
       mask.rgb_channel(im2, channel='b', threshold=(0, 100), show=show)
    elif n == 3:
       im2 = persp.warp(im, show=show)
       mask.hls_channel(im2, channel='h', threshold=(0, 90), show=show)
       mask.hls_channel(im2, channel='l', threshold=(200, 255), show=show)
       mask.hls_channel(im2, channel='s', threshold=(110, 255), show=show)
    elif n == 4:
       im2 = persp.warp(im, show=show)
       return mask.special_mask(im2, show=show)
