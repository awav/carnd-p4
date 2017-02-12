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
import cv2 as cv
import numpy as np
from calibrator import Calibrator as clb
from common import show_images

class Perspective():
    _src = np.float32([[586,455],[698,455],[1120,720],[190,720]])
    _dst = np.float32([[310,0],[1010,0],[1010,720],[310,720]])
    _perspective = None
    _perspective_inv = None
    def __new__(cls, *args, **kvargs):
        raise ValueError("You can't create `Calibration` instance")
    @classmethod
    def initialized(cls):
        return ((cls._perspective is not None) and
                (cls._perspective_inv is not None))
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
    def warp(cls, im, inverse=False, show=False):
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
            impoly = cv.polylines(im, [pts], True, (0,255,0), thickness=5)
            show_images(impoly, warped, 'original', 'warped', 'perspective transform')
        return warped
