from pprint import pprint
from typing import List
import cv2
import numpy as np

from utils import imshow
from lib import Blob, Entity, Tracker
from tuning import hyperparams, kernels


class IntrusionDetection:
    '''
    Analyse a video from a static camera in order to detect moving objects and people in the scene.
    '''

    def __init__(self, input_path: str, debug: bool = False, output_file=None):
        '''
        Initialize a the main object responsible for analysing a video.
        Parameters:
        - input_path: The path of the input video
        - debug: Log and show intermediate steps
        - output_file: Path of the text output file that will be created all the detected entities, for each frame.
        '''
        self.input_path = input_path
        self.bg = None
        self.frame_data = []
        self.debug = debug
        self.current_frame = -1
        self.tracker = Tracker(self)
        self.output_file = output_file

        if (debug):
            self.imshow = imshow
            print('Starting in debug mode')
        else:
            self.imshow = lambda *a, **b: None

        self.clahe = cv2.createCLAHE(**hyperparams['clahe'])

    def extract_background(
            self,
            frame_count: int = hyperparams['bg_extraction']['frame_count'],
            frame_skip: int = hyperparams['bg_extraction']['frame_skip']):
        """
        This function generates a clean background image from a video sequence
        Parameters:
            input_path (string): path of input video sequence
            frame_count (int): number of frames that are used to generate a clean background
            frame_skip (int): number of frames to skip before caching another frame to generate the clean background
        Returns:
            generated clean background image
        """

        cap = cv2.VideoCapture(self.input_path)
        frameNum = 0
        num_of_memorized_frames = 0
        frames = []
        while num_of_memorized_frames < frame_count:
            ret, frame = cap.read()
            if ret:
                if frameNum % frame_skip == 0:
                    frames.append(frame)
                    num_of_memorized_frames += 1
                frameNum += 1
            else:
                break
        cap.release()

        # Update height and width
        if len(frames) > 0:
            self.height = frames[0].shape[0]
            self.width = frames[0].shape[1]
            self.area_thresh = \
                self.height * self.width * hyperparams['bg_extraction']['area_thresh_perc']
        else:
            raise ValueError('Cannot initialize background! Not enough frames')

        return self._extract_background(frames=frames)

    def _extract_background(self,
                            frames,
                            alpha=hyperparams['bg_extraction']['alpha']):
        '''
        Helper function that averages out a collection of frames, obtaining an image with only static objects in the scene.
        '''
        cleanBg = alpha * np.median(np.array(frames), axis=0).astype(np.uint8) + \
            (1-alpha) * np.mean(np.array(frames), axis=0).astype(np.uint8)

        return cleanBg.astype(np.uint8)

    def start(self, breakpoints: list[int] = []):
        '''
        Run the real-time analysis for intrusion detection.
        Parameters:
        - breakpoints: if the debug flag was enabled, the image preview will stop on the selected frames' id
        '''

        # Extract background offline
        self.bg = self.extract_background()
        self.bg = self.preprocess_frame(self.bg)
        self.imshow('clean_bg', self.bg, wait=True)

        # Start capturing and analysing frames
        cap = cv2.VideoCapture(self.input_path)

        print(Entity.CSVHeader(), file=self.output_file)
        self.current_frame = -1
        while cap.isOpened():
            ret, frame = cap.read()
            self.current_frame += 1

            if not ret or frame is None:
                self.current_frame = -1
                cap.release()
                break

            if self.current_frame in breakpoints:
                cv2.waitKey(0)

            frame_copy = frame.copy()

            frame = self.preprocess_frame(frame)
            imshow('original_frame', frame)

            diff = self.subtract_background(frame)
            self.imshow('diff', diff)

            mask = self.cleanup_diff(diff)
            self.imshow('mask', mask)

            blobs = self.extract_blobs(mask)
            entities = self.tracker.track_frame(blobs, self.current_frame,
                                                frame_copy, diff)

            annotated = self.annotate_entities(frame_copy, entities)
            imshow('annotated', annotated)

            print(self.current_frame, len(entities), file=self.output_file)
            for e in entities:
                print(e.toCSV(), file=self.output_file)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()

    def preprocess_frame(
            self,
            frame,
            equalize=True,
            d=hyperparams['preprocessing']['d'],
            sigmaColor=hyperparams['preprocessing']['sigmaColor'],
            sigmaSpace=hyperparams['preprocessing']['sigmaSpace']):
        """
        Apply a defined color adjustments to a single frame.
        Parameters:
            img (ndarray): input image in BGR color space to apply adjustments on
            equalize (bool): Run clahe histogram equalization on the frame
        Returns:
            image in BGR color space after applying adjustments (noise reduction and equalization)
        """
        frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)

        if equalize:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = self.clahe.apply(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return frame

    def annotate_entities(self,
                          frame,
                          entities: list[Entity],
                          use_rect=False,
                          draw_labels=False):
        '''
        Draw the colored contours of each detected entity in the frame.
        Parameters:
        - frame: the image upon which the contours will be drawn.
        - entities: a list of entities to draw on the image.
        - use_rect: draw a rectangle instead of the contour.
        - draw_labels: annotate the image with the detected class.
        Returns:
        - The annotated image.
        NOTE: the image passed as argument is not copied and will be modified.
        '''

        for e in entities:
            x, y, w, h = e.blob.bounding_rect

            if use_rect:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), e.color,
                                      1)
            else:
                cv2.drawContours(frame, [e.blob.contour], 0, e.color, 2)

            if draw_labels:
                cv2.putText(frame, e.classification, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, e.color, 2)

        return frame

    def subtract_background(
            self,
            frame,
            threshold=hyperparams['bg_subtraction']['diff_threshold']):
        '''
        Return a mask with the difference with the background.
        Parameters:
        - frame: the image from which the previously detected background will be subtracted
        - threshold: int between 1 and 255
        Returns:
        - An image where each pixel intensity indicates how much it differs from the original background.
        '''
        frame = frame.astype(np.int16)
        sub = np.subtract(self.bg, frame)
        abs = np.abs(sub)
        channels_sum = (np.sum(abs, axis=2) / 3).astype(np.uint8)

        # Compute an initial mask, by applying a threshold to the diff
        mask = cv2.threshold(channels_sum, threshold, 255,
                             cv2.THRESH_BINARY)[1].astype(np.uint8)
        mask = mask.astype(np.uint8)
        return mask

    def cleanup_diff(
            self,
            diff,
            padding_size=hyperparams['mask_cleaning']['padding_size']):
        '''
        Perform morphological operations to reduce noise, filter out small blobs, and improve the contour of the detected objects.
        Parameters:
        - diff: difference between the frame and the detected background.
        '''

        channels_sum = cv2.copyMakeBorder(diff,
                                          padding_size,
                                          padding_size,
                                          padding_size,
                                          padding_size,
                                          cv2.BORDER_CONSTANT,
                                          value=0)

        # Filter out small blobs
        contours = cv2.findContours(channels_sum, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)[0]
        cleanedMask = np.zeros(channels_sum.shape[0:2]).astype(np.uint8)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area >= self.area_thresh:
                cv2.drawContours(cleanedMask, [contours[i]], -1, 255,
                                 cv2.FILLED)

        # Apply morphological operators to the mask to improve the quality
        cleanedMask = cv2.morphologyEx(cleanedMask,
                                       cv2.MORPH_CLOSE,
                                       kernels['small'],
                                       iterations=3)

        cleanedMask = cv2.dilate(cleanedMask, 
                                 kernels['dilation'], 
                                 iterations=1)
        
        cleanedMask = cv2.morphologyEx(cleanedMask,
                                       cv2.MORPH_CLOSE,
                                       kernels['big'],
                                       iterations=4)

        cleanedMask = cv2.morphologyEx(cleanedMask,
                                       cv2.MORPH_OPEN,
                                       kernels['small'],
                                       iterations=1)

        cv2.waitKey(0)
        # Remove padding from mask
        cleanedMask = cleanedMask[padding_size:-padding_size,
                                  padding_size:-padding_size]

        return cleanedMask

    def extract_blobs(self, mask) -> list[Blob]:
        '''
        Extract Blob objects from the provided mask, in order to continue with further analysis.
        Parameters:
        - mask: the mask of detected objects.
        Returns:
        - a list of Blob objects
        '''
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]

        return [Blob(c) for c in contours]
