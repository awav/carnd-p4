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
import calibrator
import perspective
import mask
import common

import importlib
importlib.reload(mask)
importlib.reload(common)

from calibrator import Calibrator as clb
from perspective import Perspective as prsp
from mask import CustomMask as custom_mask
from common import show_images

from moviepy.editor import VideoFileClip

#from abc import ABCMeta, abstractmethod
#class IPipeline(metaclass=ABCMeta):
#    @abstractmethod
#    def process(self, arg):
#        pass

class Lane():
    def __init__(self, points, start=0, end=719, num=720):
        yp, xp = points
        coeff = np.polyfit(yp, xp, 2)
        y = np.linspace(start, end, num)
        self.a = coeff[0]
        self.b = coeff[1]
        self.c = coeff[2]
        self.line = self.a*(y**2) + self.b*y + self.c
        self.coeff = coeff
        self.y = y
    def cross(self, y=0):
        return self.a * (y**2) + self.b * y + self.c
    def curvature(self, y=0):
        x_cm = 3.7 / 700
        y_cm = 30 / 720
        a, b, _ = np.polyfit(self.y * y_cm, self.line * x_cm, 2)
        return (1 + ((a * y * y_cm * 2) + b)**2)**1.5 / (a * 2)
    @staticmethod
    def fill_lanes(im, llane, rlane, fill_color=(0,255,255), show=False):
        y = llane.y 
        lp = np.array([np.transpose(np.vstack([llane.line, y]))])
        rp = np.array([np.flipud(np.transpose(np.vstack([rlane.line, y])))])
        points = np.hstack([lp, rp])
        fill = np.zeros_like(im)
        cv.fillPoly(fill, np.int32([points]), fill_color)
        if show == True:
            cpy = im.copy()
            cpy = cv.addWeighted(cpy, 1, fill, 0.2, 0)
            show_images(im, cpy, 'origin', 'filled warp', 'Fill lanes')
        return fill
    @staticmethod
    def center(llane, rlane, y=0):
        return (llane.cross(y) + rlane.cross(y)) // 2

class FrameLanePipeline():
    def __init__(self):
        assert(clb.initialized())
        assert(prsp.initialized())
        self.left_line = None
        self.right_line = None
    def process(self, frame, show=False):
        """
        Process function takes RGB image frame.
        It undistorts and warpes frame to eye bird view. 
        """
        im = clb.undistort(frame);
        warped = prsp.warp(im)
        masked = custom_mask.apply(warped, show=show)
        lp, rp = self._find_lane_points(masked, num_strips=10, radius=70, show=show)
        height = masked.shape[0]
        llane = Lane(lp, end=height-1, num=height)
        rlane = Lane(rp, end=height-1, num=height)
        center = Lane.center(llane, rlane, y=masked.shape[0]-1)
        lcurv = llane.curvature(height // 2)
        rcurv = rlane.curvature(height // 2)
        fill = Lane.fill_lanes(warped, llane, rlane, show=show)
        return self._final_frame(im, fill, lcurv, rcurv, show=show)
    def _final_frame(self, im, fill, lcurv, rcurv,
                     font=cv.FONT_HERSHEY_DUPLEX,
                     scale_font=1, color_font=(255,0,0), 
                     show=False):
        fill = prsp.warp(fill, inverse=True)
        out = im.copy()
        out = cv.addWeighted(out, 0.7, fill, 0.5, 0)
        xtxt = 50
        lcurv_text = 'Left curvature: {0:.01f}m'.format(lcurv)
        rcurv_text = 'Right curvature: {0:.01f}m'.format(lcurv)
        out = cv.putText(out, lcurv_text, (xtxt, 30), font, scale_font, color_font)
        out = cv.putText(out, rcurv_text, (xtxt, 60), font, scale_font, color_font)
        if show == True:
            show_images(im, out, 'origin', 'lanes', 'Lanes detected')
        return out
    def _find_peaks(self, strip_im):
        mid = np.int32(strip_im.shape[1] * .5)
        lx = strip_im[:,:mid].sum(axis=0).argmax()
        rx = strip_im[:,mid:].sum(axis=0).argmax() + mid
        return lx, rx
    def _nonzero_points(self, nonz, left=((0,200),(720,600)), right=((0,800), (720,1200))):
        # lty => left top y
        # ltx => left top x
        # lby => left bottom y
        # lbx => left bottom x
        (lty, ltx), (lby, lbx) = left
        (rty, rtx), (rby, rbx) = right
        l = (nonz[0] >= lty) & (nonz[0] < lby) & (nonz[1] > ltx) & (nonz[1] <= lbx)
        r = (nonz[0] >= rty) & (nonz[0] < rby) & (nonz[1] > rtx) & (nonz[1] <= rbx)
        nonz_l = nonz[:, l]
        nonz_r = nonz[:, r]
        return (nonz_l[0], nonz_l[1]), (nonz_r[0], nonz_r[1])
    def _find_lane_points(self, im, num_strips=10, radius=70, show=False):
        strip_height = im.shape[0] // num_strips
        heights = [None] * num_strips
        strips = [None] * num_strips
        for i in range(num_strips):
            s, e = i * strip_height, (i+1) * strip_height
            heights[-i-1] = (s, e)
            strips[-i-1] = im[s:e, :]
        ly, lx = [[]] * num_strips, [[]] * num_strips
        ry, rx = [[]] * num_strips, [[]] * num_strips
        nonzeros = np.array(im.nonzero())
        peaks = [None] * num_strips
        if show == True:
            cpy = im.copy()
        lpeak, rpeak = 0, 0
        for i in range(num_strips):
            if i == 0:
                lpeak, rpeak = self._find_peaks(im)
                rad = 100
            else:
                lpeak, rpeak = self._find_peaks(strips[i])
                rad = radius
            peaks[i] = (lpeak, rpeak)
            lbox = ((heights[i][0], lpeak - rad), (heights[i][1], lpeak + rad))
            rbox = ((heights[i][0], rpeak - rad), (heights[i][1], rpeak + rad))
            if show == True:
                top, bot = lbox[0], lbox[1]
                cpy = cv.rectangle(cpy, (top[1], top[0]), (bot[1], bot[0]), (i+1,0,0), 2)
                top, bot = rbox[0], rbox[1]
                cpy = cv.rectangle(cpy, (top[1], top[0]), (bot[1], bot[0]), (i+1,0,0), 2)
            (ly[i], lx[i]), (ry[i], rx[i]) = self._nonzero_points(nonzeros, left=lbox, right=rbox)
        left = (np.concatenate(ly), np.concatenate(lx))
        right = (np.concatenate(ry), np.concatenate(rx))
        if show == True:
            filtered = np.zeros(im.shape[:2])
            filtered[left[0],left[1]] = 1
            filtered[right[0],right[1]] = 1
            show_images(filtered, cpy, 'filtered', 'regions', 'Found points', cmap1='gray')
        return left, right

class VideoLanePipeline():
    def __init__(self, cls=FrameLanePipeline):
        self.handler = cls().process
    def process(self, video_file, output_file):
        """
        Process function takes `video_file` argument as path to video file.
        It loads video and process it frame by frame.
        """
        video = VideoFileClip(video_file)
        out = video.fl_image(self.handler)
        out.write_videofile(output_file, audio=False)

def argmedian(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l,r = len(a)/2 -1, len(a)/2
        left = np.partition(a, l)[l]
        right = np.partition(a, r)[r]
        return [np.where(a == left)[0][0], np.where(a==right)[0][0]]

