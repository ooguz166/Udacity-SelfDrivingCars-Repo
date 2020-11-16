# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os,math
import glob
import os.path

class lineDetector():
    def __init__(self, inputPath, outputPath):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 10
        self.min_line_length = 50
        self.max_line_gap = 1
        self.kernel=7
        self.yThres=329
        self.color=[255, 0, 0]
        self.colorb = [0, 255, 0]
        self.thickness=10
        self.eps=0.00001
        self.prevX1=[]
        self.prevX2=[]
        self.count=0
    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def canny(self,img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self,img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self,img):
        # defining a blank mask to start with
        mask = np.zeros_like(img)
        vert=np.array([[(40,img.shape[0]),(460, 310), (490, 320), (img.shape[1]-20,img.shape[0])]], dtype=np.int32)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vert, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def get_mean_xline_points(self,img, lines):
        """
        The idea of this function is as follows: if you can find the mean m and b values of the lines, you can approximate the
        y=xm+b equation.

        Since we limit the region with roi y value we know the y axis values thus by reversing the equation (x= (y-b)/m) we
        can calculate the x values of the mean lines for both left and right parts

        """
        # Containers inits
        positive_cord_list = []
        negative_cord_list = []

        pos_slope = []
        neg_slope = []

        # Collect the b values of the left and right lines
        positive_intercept = []
        negative_intercept = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate slope of the line

                slope = np.divide((float(y2) - float(y1)), (float(x2) - float(x1)))


                # Positive slope with b value
                if slope >= 0:
                    if math.isnan(float(y1) - float(x1) * slope):
                        continue
                    positive_coord = [x1, y1, x2, y2]
                    pos_slope.append(slope)
                    positive_cord_list.append(positive_coord)
                    positive_intercept.append(float(y1) - float(x1) * slope)

                # Negative Slope with b value
                elif (slope < 0):
                    if math.isnan(float(y1) - float(x1) * slope):
                        continue
                    negative_coord = [x1, y1, x2, y2]
                    neg_slope.append(slope)
                    negative_cord_list.append(negative_coord)
                    negative_intercept.append(float(y1) - float(x1) * slope)

        # Algorithm some times fails to get lines at given frame, use previous frame's
        if  pos_slope:
            self.prevslope=pos_slope
            self.prevPosIntercept=positive_intercept
            x1_pos, x2_pos = self.get_xpoints(img, pos_slope, positive_intercept)
            x1_neg, x2_neg = self.get_xpoints(img, neg_slope, negative_intercept)
        else:
            x1_pos, x2_pos = self.get_xpoints(img, self.prevslope, self.prevPosIntercept)
            x1_neg, x2_neg = self.get_xpoints(img, neg_slope, negative_intercept)




        return x1_pos, x2_pos, x1_neg, x2_neg

    def get_xpoints(self,img, slope, b):
        # For each line find the start and end points using the m and b
        # By assuming the we detect the lines in very good manner, y value can b the minimum y value
        # However assumption tends to fail thus limitting y with roi region.
        # Note that if the road is not flat this apporach leads to short lines

        # x = (y-b)/m:
        x1 = np.divide((img.shape[0] - np.mean(b)), np.mean(slope) + self.eps)
        x2 = np.divide((self.yThres - np.mean(b)), np.mean(slope) + self.eps)

        return x1, x2

    def hough_lines(self,img):
        lines = cv2.HoughLinesP(img, self.rho, self.theta, self.threshold, np.array([]), minLineLength=self.min_line_length,maxLineGap=self.max_line_gap)
        return lines

    def weighted_img(self,img, initial_img, a=0.8, b=1., y=0.):

        return cv2.addWeighted(initial_img, a, img, b, y)

    def line_detect_image_pipeline(self,image):
        # Gray Conversion
        gray_img = self.grayscale(image)
        # Blurring for Canny
        blur_gray_img = self.gaussian_blur(gray_img, self.kernel)
        # In order to find proper thresholds use otsu threshold as a center
        otsu_thereshold, _ = cv2.threshold(blur_gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Canny thresholds are around the otsu threshold
        thr_1 = int(max(0, (0.37) * otsu_thereshold))
        thr_2 = int(min(255, (1.1) * otsu_thereshold))
        edge_img = self.canny(blur_gray_img, thr_1, thr_2)

        masked_img = self.region_of_interest(edge_img)
        # plt.imsave("./dummy/"+str(self.count)+".jpg",masked_img)

        # Run Hough on edge detected image

        lines = cv2.HoughLinesP(masked_img, self.rho, self.theta, self.threshold, np.array([]), self.min_line_length, self.max_line_gap)

        x1_pos, x2_pos, x1_neg, x2_neg = self.get_mean_xline_points(image, lines)

        if self.count ==0:
            self.prevX1=x1_neg
            self.prevX2=x2_neg
        self.count = self.count + 1


        if math.isnan(x2_neg):

            img_lines = np.zeros((masked_img.shape[0], masked_img.shape[1], 3), dtype=np.uint8)

            cv2.line(img_lines, (int(x1_pos), int(masked_img.shape[0])), (int(x2_pos), int(self.yThres)), self.color,
                     self.thickness)
            cv2.line(img_lines, (int(self.prevX1), int(masked_img.shape[0])), (int(self.prevX2), int(self.yThres)), self.colorb,
                     self.thickness)
        else:

            img_lines = np.zeros((masked_img.shape[0], masked_img.shape[1], 3), dtype=np.uint8)

            cv2.line(img_lines, (int(x1_pos), int(masked_img.shape[0])), (int(x2_pos), int(self.yThres)), self.color,
                     self.thickness)
            cv2.line(img_lines, (int(x1_neg), int(masked_img.shape[0])), (int(x2_neg), int(self.yThres)), self.colorb,
                     self.thickness)





        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(img_lines, 0.8, image, 1, 0)

        return lines_edges

    def line_detect_vid_pipeline(self,videoPath,vidName,savePath):
        cap = cv2.VideoCapture(videoPath)
        line_frames= []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                line_frame=self.line_detect_image_pipeline(image=frame)
                line_frames.append(line_frame)

            # Break the loop
            else:
                break
        cap.release()

        out = cv2.VideoWriter(savePath+vidName+'_output_lines_video.avi', cv2.VideoWriter_fourcc(*"mp4v"),60, (960, 540))

        for frm in (line_frames):
            out.write(frm)

        out.release()

    def process_folder_images(self):
        for file in os.listdir(self.inputPath):
            image = mpimg.imread(os.path.join(self.inputPath, file))

            outputPath = (os.path.join(self.outputPath, file))
            lane_image=self.line_detect_image_pipeline(image)
            plt.figure()
            plt.imsave(outputPath,lane_image)


if __name__ == '__main__':
    processImages = False

    if processImages:
            detector = lineDetector(outputPath="/test_output_images/", inputPath="test_images/")
            detector.process_folder_images()
    else:
        detector = lineDetector(inputPath="", outputPath="")

        dir = './test_videos/'
        videos = glob.glob(os.path.join(dir, '*.mp4'))
        for vid in videos:
            base = os.path.basename(vid)
            if base=='challenge.mp4':
                continue
            vidname=os.path.splitext(base)
            detector.line_detect_vid_pipeline(videoPath=vid,vidName=vidname[0],savePath="test_videos_output/")
