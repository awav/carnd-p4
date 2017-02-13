## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

The goals of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


Project files
---

* `calibrator.py` calculates calibration of camera and undistortes video frames or just images.
* `mask.py` gives different mask options for input images.
* `perspective.py` keeps _src_, _dst_ points and perspective transform matrices - standard and inverse a s well.
* sperspectivs.py` keeps _src_, _dst_ points and perspective transform matrices - standard and inverse a s well.
* sperspective.py` keeps _src_, _dst_ points and perspective transform matrices - standard and inverse a s well.
* `common.py` auxiliary functions that help to debug and produce some visual results.
* `pipeline.py` this module contains classes for frame and video handling.

### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#### Criteria: provide an example of a distortion-corrected image.

To solve camera calibration problem I wrote class that searches chessboard images in target folder by regexp patter and saves found filenames.

```python
## calibrator.py
def find_pictures(cls, pattern='.*\.jpg', directory='.'):
    if not os.path.isdir(directory):
        ## TOOD: log this message
        return []
    regex = re.compile(pattern)
    files = []
    for filename in os.listdir(directory):
        if re.match(string=filename, pattern=regex) is None:
            continue
        files.append(os.path.join(directory, filename))
    if files != []:
        cls._files = files
    return files
```

Then `Calibrator` uses this list to load chessboard images and detects N by M corner coordinates. N is a number of corners in y-axis and M is a number of corner on x-axis. Then using OpenCV `calibrateCamera` method, found points and groundtruth coordinates for corners also known as `objpoints`, the calibrator computes **distortion matrix** and **camera matrix**. This is crusial components which `Calibrator` saves once they were calculated.

```python
## calibrator.py
def calibrate_camera(cls, nx, ny, show=False):
    assert(len(cls._files) != 0)
    objs = np.zeros((nx * ny, 3), dtype=np.float32)
    objs[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)
    n = len(cls._files)
    objpoints = [objs] * n
    imgpoints = []
    for i in range(n):
        img = cv.imread(cls._files[i], cv.IMREAD_COLOR)
        if img is None:
            print('{0} is not an image'.format(cls._files[i]), file=sys.stderr)
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            if show == True:
                cv.drawChessboardCorners(img, (nx, ny), corners, ret)
                fig = plt.figure(0)
                fig.canvas.set_window_title('calibrate image #{0}'.format(i))
                plt.imshow(img)
                plt.show()
        elif show == True:
            fig = plt.figure(0)
            fig.canvas.set_window_title('FAILED image #{0}'.format(i))
            plt.imshow(img)
            plt.show()
    nimg = len(imgpoints)
    if nimg > 0:
        shape = img.shape[1], img.shape[0]
        ret, mtx, dist, _rv, _tv = cv.calibrateCamera(
                objpoints[:nimg], imgpoints,
                imageSize=shape, cameraMatrix=None, distCoeffs=None)
        if ret:
            cls._mtx = mtx
            cls._dist = dist
```

**distortion matrix** and **camera matrix** are arguments for OpenCV `undistort` function, which makes process of true image reconstruction is much easier. `Calibrator` has special wrapper for it:

```python
## calibrator.py
def undistort(cls, im, show=False):
     assert(cls._mtx is not None)
     assert(cls._dist is not None)
     undist_im = cv.undistort(im, cls._mtx, cls._dist, None, cls._mtx)
     if show == True:
         fig, ax = plt.subplots(ncols=2, squeeze=True)
         fig.canvas.set_window_title('Distorted/Undistored Image')
         ax[0].imshow(im)
         ax[1].imshow(undist_im)
         ax[0].set_title('distorted', color='r')
         ax[1].set_title('undistorted', color='b')
         fig.tight_layout()
         fig.show()
     return undist_im
```

Looks like everyting is ready to undistort the test image. The result you can see in a screenshot below. You can noticethat origin frame has curved billboard with highway markings, but on corrected image it looks straight.

```python
from calibrator import Calibrator as clb
clb.find_pictures(directory="camera_cal", pattern=".*\.jpg")
n, m = 9, 6
clb.calibrate_camera(n, m)
im = common.load_image("vehicles.png")
undistored = clb.undistort(im, show=True)
```

![Alt text](project/undistorted.png)


### Apply a perspective transform to rectify binary image ("birds-eye view").
#### Criteria: Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I decided to swap masking and perspective transformation steps, because my experiments showed that applying masks after perspective step reduces noise a lot.

I created another one module that takes responsiblity for managining perspective transformations. All logic is in `Perspective` class in `perspective.py` file. The core principal transformation is transformation matrix. The OpenCV provides valuable function `getPerspectiveTransform`, so that we don't need to worry about out own implementation of perspective transformation, despite that it would quite interesting. The only what we need is source box destination box points. The source rectangle is a region in which we are interested and we would like to see it in bird-eye view. Passing corrdinates of box vertices to `getPerspectiveTransform`, we can get two transformation matrices: _standard_ and _inverse_. The bird-eye view is core principle of advanced lane finding algorithm. Once image is converted to bird-eye view and lanes are detected, for visualization of found lanes we need to transform it back, so, that is a place where _inverse_ matrix plays an important role. The `inverse=[True|False]` argument in `warp` method of `Perspective` class gives you control of what type of matrix you will use.

```python
## perspective.py
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
        impoly = cv.polylines(im, [pts], True, (0,255,0))
        show_images(impoly, warped, 'original', 'warped', 'perspective transform')
    return warped
```

Here is coordinates of vertices which are set up by default:

```python
## perspective.py
class Perspective():
    _src = np.float32([[586,455],[698,455],[1120,720],[190,720]])
    _dst = np.float32([[310,0],[1010,0],[1010,720],[310,720]])
    _perspective = None
    _perspective_inv = None
```

This is an example of perspective transformed image by `Perspective.warp` method:

```python
from perspective import Perspective as prsp
prsp.find_perspective()
im = common.load_image("./test_images/test1.jpg")
warped = prsp.warp(im, inverse=False, show=True)
```

![Alt text](project/warped.png)

### Use color transforms, gradients, etc., to create a thresholded binary image.
#### Criteria: Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I devoted plenty of time to find best configuration for masking. The result I have gotten is that mask of level and saturation channels with thresholds `[100, 255)` and `[200, 255)` respectively gave me best results. I also found out that green channel with `[200, 255)` and cut magnitude of Sobel can add more stability, especially for top half of the image where all objects are zoomed in and intensities are blurred.

I created a module `mask.py` to perform different types of masks. There are 3 major classes: `CustomMask`, `ColorThresholder` and `SobelMask`. `CustomMask` provides final mask solution for my pipeline.

```python
## mask.py

ass CustomMask():
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
```

Below you can find an example of applied `CustomMask`:

```python
from mask import CustomMask as custom_mask
custom_mask.apply(warped)
```

Binary colorful mask. It has saturation, level channes of HLS, and also green channel of RGB, plus thresholded magnitude of sobel:
![Alt text](project/colorful_masking.png)

Binary representation:
![Alt text](project/binary_masking.png)


