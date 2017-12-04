**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_images/car_notcar.jpg
[image3]: ./report_images/car_hog1.jpg
[image7]: ./report_images/car_nhog1.jpg
[image10]: ./report_images/400_500_1.jpg
[image11]: ./report_images/400_550_1_3.jpg
[image12]: ./report_images/400_600_1_7.jpg
[image13]: ./report_images/450_700_2_5.jpg
[image14]: ./report_images/pipeline0.jpg
[image15]: ./report_images/pipeline1.jpg
[image16]: ./report_images/pipeline2.jpg
[image17]: ./report_images/pipeline3.jpg

[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section called SVM Classifier of the IPython notebook (the function executing it is called `get_hog_features(...)` and located in the file `feature_extraction.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for a random car image from the provided datased:

![alt text][image3]

The same for a non-car image:

![alt text][image7]

#### 2. Explain how you settled on your final choice of HOG parameters.

The configuration above did reasonably well for classification, so I have decided to choose it. The intuition behind this particular choice is the following:

* The parameter `orientations` shouldn't be very large which means we tend to describe object of boxed shape with angles varying not more than +-40 degrees (360/9 = 40). Cars satisfy this property.
* The other two parameters (`pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`) is a good choice for 64x64 image allowing to get reasonably enough amount of features from a given picture. I will provide a very simplified explanation. The car shape changes about 1-3 times (if we look from behind or from any side) in horisontal direction and about 1-2 times in vertical direction. This means that for good description of a car shape we need about 2-3 boxes per one shape change in order to describe these changes with reasonable amount of details. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear SVM classifier using HOG features and color features and got 99.6% on the test set (section "SVM Classifier" in the Jupyter Notebook file), which was enough for reliable car detection in a video stream. The disadvantage though is the computational complexity, i.e. in order to classify if the image belongs to one of two classes, the classifier should analyse its histogram, pixels and hog features.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Because of the effect of perspective the car appears small near the middle of the image and large near its edges. Therefore I searched in the following four regions trying to take this into account (function `pipeline(...)` lines #8 - #39).

* `scale = 1`

![alt text][image10]

* `scale = 1.3`

![alt text][image11]

* `scale = 1.7`

![alt text][image12]

* `scale = 2.5`

![alt text][image13]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 different scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. The images at each stage of working pipeline are provided in the next section. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Assuming that each blob corresponded to a vehicle, I constructed bounding boxes to cover the area of each blob detected. The code with detections is in the function `find_cars_v2()`, and filtering false positives is in `add_heat()` and `apply_threshold()` which is the part of the function `find_cars_v2()`. 

In addition I do [exponential smoothing](https://www.wikiwand.com/en/Exponential_smoothing) of a heatmap between frames. This helps to make box movements more smooth:

```python
  heat_m = np.uint8(0.7*add_heat(heat, list(boxes)) + (1-0.7)*heat_m)
```

Here's an example result showing how the pipeline works. Left image is the SVM detection result, then its heatmap, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last picture:

![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project I studied computer vision approach to solving vehicle detection problem. There are some advantages and disadvantages of current solution:

* It is quite simple from conceptual point of view. This means that developer understands every single bit in the algorithm, whereas, for example, in deep learning approach, the algorithm is treated like a 'black box'. So it is behaviour is predictable.

* It is very demanding to computational resources. Yes, it works in the project video, but the algorithm like this should solve the same problem in real-time if we want to use it in real car,

* My SVM classifier can only detect cars and not cars. If there are more classes of objects to detect, the problem becomes harder and computational complexity increases. However improving a classifier (increasing the size of datased, use a different type, i.e. neural network) is one of the ways to make this approach more robust. 

There are different solutions based on deep learning, for example YOLA ([their website](https://pjreddie.com/darknet/yolo/), [git hub](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection) or [paper](https://arxiv.org/abs/1506.02640)).YOLA allows getting about 40-60 fps real-time and solving exactly the same problem, but not for cars only. Looks pretty impressive even in the case of high number of objects in the frame. I think this method can be a good alternative to the implemented one. 

