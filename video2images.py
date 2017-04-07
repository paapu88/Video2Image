"""
python3 ~/PycharmProjects/Rekkari/video2images.py "Rekisteri_4_short.mp4" 0 2700 25

Functions related to read and write videos and images in opencv
filename firstFrame lastFrame stride
"""


import cv2
import numpy as np
import sys
import time


class VideoIO():
    def __init__(self, videoFileName=None, fps=24, first_frame=0, last_frame=None, stride=1, colorChange=cv2.COLOR_RGB2GRAY):

        self.fps = fps
        self.videoFileName = videoFileName
        #self.cap = cv2.VideoCapture(*args)
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.stride = stride
        self.colorChange=colorChange
        self.n_frames=None
        self.frames=None
        print ("INIT OK")

    def changeColor(self, imageIn):
        if self.colorChange=='mok.COLOR_YUV2GRAY_420':
            N=imageIn[0]*imageIn[1]
            print("N",N, N.shape)
            sys.exit()
        else:
            return cv2.cvtColor(imageIn, self.colorChange)

    def readVideoFrames(self, videoFileName=None):
        """ read in frames and convert to desired format"""
        if videoFileName is None:
            videoFileName = self.videoFileName
        if self.first_frame is None:
            first_frame = 0
        else:
            first_frame = self.first_frame
        last_frame = self.last_frame

        print ("starting reading: filename", videoFileName)
        cap = cv2.VideoCapture(videoFileName)
        while not cap.isOpened():
            cap = cv2.VideoCapture(videoFileName)
            cv2.waitKey(1000)
            print("Wait for the header")

        print("self:", last_frame)
        if last_frame is None:
            last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("LAST NR", )
        self.frames = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        count = self.first_frame
        while True:
            flag, frame = cap.read()
            print("FRAME:", frame.shape)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if flag:
                gray = self.changeColor(imageIn=frame)
                #gray = cv2.cvtColor(frame, self.colorChange)
                if count >= self.first_frame + stride:
                    self.frames.append(gray)
                    count=self.first_frame
                    print (str(pos_frame)+" frames")
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                print("frame is not ready")
                # It is better to wait for a while for next frame to be ready
                cv2.waitKey(1000)

            count = count + 1
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == last_frame+1:
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break



        self.n_frames=last_frame-first_frame+1
        cap.release()
        print("finished reading")

    def setColorChange(self, colorChange):
        self.colorChange = colorChange


    def writeAllImages(self, format='jpg', prefix=''):
        """Write given frames to disk as images one-by-one"""

        for i, frame in enumerate(self.frames):
            cv2.imwrite(prefix+'.'+str(i+self.first_frame)+'.'+format, frame)



    def backgroundSubtractorMOG2(self):
        """
        http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/ \
        py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction
        """

        fgbg = cv2.createBackgroundSubtractorMOG2()

        for frame in self.frames:
            fgmask = fgbg.apply(frame)
            #self.toContours(fgmask, frame)
            cv2.imshow('frame',fgmask)
            while True:
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

        cv2.destroyAllWindows()


    def toContours(self, mask, frame):
        """
        after background subtarctorm blur edges and get contours
        """
        sys.path.append('/home/mka/PycharmProjects/')
        from Image2Letters.initialCharacterRegions import InitialCharacterRegions
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE, kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel2)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        regions=[]
        for contour in contours:
            regions.append(cv2.boundingRect(contour))
        dummy = InitialCharacterRegions()
        dummy.showAllRectangles(clone=frame.copy(), regions=regions)


    def backgroundSubtractorKNN(self):
        """
        http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/ \
        py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction
        """

        fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

        for frame in self.frames:
            #fgmask = cv2.bitwise_not(fgbg.apply(frame))
            fgmask = fgbg.apply(frame)
            self.getContourRectangles(fgmask, frame)
            cv2.imshow('frame',frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def camShift(self, initFrame, initRectangle):

        # setup initial location of window
        # x,y,width,height
        (r,h,c,w) = initRectangle
        track_window = initRectangle

        # set up the ROI for tracking
        roi = self.frames[initFrame][r:r+h, c:c+w]
        #hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        roi_hist = cv2.calcHist(roi,[0], mask=None, histSize=[255],ranges=[0,255])
        #cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        for frame in self.frames[initFrame:]:

            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            dst = cv2.calcBackProject(roi,[0],roi_hist,[0,255],1)

            # apply camshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,5)
            cv2.imshow('img2',img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)



    def backgroundSubtractorSimple(self, minArea = None):
        """assume constant background"""


        background = self.frames[0]

        for frame in self.frames:
            frameDelta = cv2.absdiff(background, frame)
            #thresh = cv2.adaptiveThreshold(frameDelta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            #                                cv2.THRESH_BINARY, 11, 2)
            thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
            # fill holes in thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            self.getContourRectangles(thresh, frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
            #while True:
            #    k = cv2.waitKey(30) & 0xff
            #    if k == 27:
            #        break

        cv2.destroyAllWindows()

    def getContourRectangles(self, thresh, frame, minArea=None):

        # assume a car is at least 1% of the image area
        if minArea is None:
            minArea = self.frames[0].shape[0] * self.frames[0].shape[1] * 0.01

        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        for contour in contours:
            if cv2.contourArea(contour) < minArea:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)






if __name__ == '__main__':
    import sys
    videoFileName=sys.argv[1]
    first_frame=int(sys.argv[2])
    last_frame=int(sys.argv[3])
    stride=int(sys.argv[4])
    
    video2images = VideoIO(videoFileName=videoFileName,
                           first_frame=first_frame,
                           last_frame=last_frame,
                           stride=stride,
                           colorChange=cv2.COLOR_RGB2GRAY)
    video2images.readVideoFrames(videoFileName=videoFileName)
    video2images.camShift(0, (1000,650,50,20))
    #video2images.backgroundSubtractorSimple()
    #video2images.backgroundSubtractorKNN()
    #video2images.backgroundSubtractorMOG2()
    #video2images.writeAllImages(prefix=videoFileName)




