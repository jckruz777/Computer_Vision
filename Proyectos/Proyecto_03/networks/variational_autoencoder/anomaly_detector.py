import cv2
import numpy as np

class AnomalyDetector:
    def __init__(self, anomaly_treshold = 0.4):
        self._anomaly_treshold = anomaly_treshold

    def _segment(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, tresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(tresh, cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        return cv2.watershed(img,markers)

    def evaluate(self, reconstruction_error, ssim, orig, rec, dataset_id):
        img_width = 200 if dataset_id==1 else 360
        img_height = 200 if dataset_id==1 else 290
        if ssim < self._anomaly_treshold and reconstruction_error > (self._anomaly_treshold * 100):
            print("Anomaly detected!!")

            # Get the anomaly region
            rec = cv2.cvtColor(rec.astype('uint8'), cv2.COLOR_RGB2BGR)
            sub = cv2.subtract(cv2.resize(orig, (img_width, img_height)), cv2.resize(rec, (img_width, img_height)))
            markers = self._segment(sub)
            anomalies = cv2.resize(orig, (img_width, img_height))

            cv2.imshow('Original', cv2.resize(orig, (img_width, img_height)))
            cv2.imshow('Reconstruction', np.array(cv2.resize(rec, (img_width, img_height)), dtype = np.uint8 ))
            anomalies[markers == -1] = [255,0,0]
            cv2.imshow('Diff', cv2.resize(anomalies, (img_width, img_height)))
            cv2.imshow('Subtraction', sub)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
