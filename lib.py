from multiprocessing.sharedctypes import Value
from pprint import pprint
import cv2
import numpy as np
from functools import cached_property
from itertools import count
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from tuning import hyperparams

# Used to assign unique ids to the entity.
counter = count()


class Tracker:
    '''
    Singleton in charge of tracking objects moving in the scenes, by associating newly detected blobs
    with the previously detected entities.
    '''

    def __init__(self, context):
        '''
        Parameters:
        - context: The parent IntrusionDetection object, which some global scene information
        '''
        self.context = context
        self.entities_by_frame = defaultdict(list)

    def get_entities(self, frame_index) -> list['Entity']:
        '''
        Retrieve all entities detected in a particular frame
        '''
        return self.entities_by_frame[frame_index]

    def track_frame(self, blobs: list['Blob'], frame_index: int, frame,
                    bg_diff):
        '''
        Tries to associate newly detected blobs with previous entities.
        In doing so, it updates the classification scores by running an heuristic function.
        Parameters:
        - blobs: A list of the newly detected Blob objects
        - frame_index: the index of the current frame
        - frame: The current frame image
        - bg_diff: The difference between the current frame and the detected background
        '''

        current_entities = []
        for b in blobs:
            for e in self.entities_by_frame[frame_index - 1]:
                if b.similar(e.blob) and e not in current_entities:
                    e.update(b)
                    current_entities.append(e)
                    break
            else:
                current_entities.append(Entity(b, frame_index))

        for e in current_entities:
            e.compute_scores(frame, bg_diff, self.context.bg)

        self.entities_by_frame[frame_index] = current_entities
        return self.entities_by_frame[frame_index]


class Entity(object):
    '''
    Track the evolution of a single blob over time.
    '''

    @property
    def blob(self) -> 'Blob':
        '''
        The latest blob object associated with this entity.
        '''
        return self._blob

    @property
    def speedX(self) -> float:
        '''
        How much the blob's centroid has moved along the X axis.
        '''
        return self._speedX

    @property
    def speedY(self) -> float:
        '''
        How much the blob's centroid has moved along the Y axis.
        '''
        return self._speedY

    @property
    def color(self):
        '''
        A utility function used to label the detected object's class with color (BGR).
        Returns:
        - An RGB triplet: Blue for a person class, Red for a false object, Green for a true object.
        '''
        if self.classification == 'person':
            return (255, 0, 0)
        if self.classification == 'true_o':
            return (0, 255, 0)
        if self.classification == 'false_o':
            return (0, 0, 255)
        else:
            return ValueError('Class not found')

    @property
    def id(self) -> str:
        '''
        The unique id associated with the entity.
        The id is in the format #FXXXX_YYYY where XXXX corresponds to the frame index where the entity
        is initially detected, and YYYY is a progressive unique integer.
        '''
        return self._id

    def __init__(self, blob: 'Blob', frameNumber):
        self._id = f'#F{frameNumber:04}_{counter.__next__():04}'
        # print('Creating Entity', self._id)
        self._blob = blob
        self._prev_blob = blob
        self._speedX = 0.0
        self._speedY = 0.0

        self._personScore = 0.0
        self._objectScore = 0.0

    @property
    def classification(self) -> str:
        '''
        Returns the detected class of the entity.
        '''

        if self._personScore > 0:
            return 'person'
        elif self._objectScore > 0:
            return 'true_o'
        else:
            return 'false_o'

    @staticmethod
    def CSVHeader() -> str:
        return 'id; area; perimeter; aspect; centroidX; centroidY; boundingRectX; boundingRectY; boundingRectH; boundingRectW; classification'

    def toCSV(self, sep='; ') -> str:
        '''
        Return all the geometric properties related to the entity's shape.
        '''
        return sep.join([self.id] + list(
            map(lambda x: str(int(1000 * x) / (1000.0)), [
                self.blob.area,
                self.blob.perimeter,
                self.blob.aspect,
                self.blob.centroid[0],
                self.blob.centroid[1],
            ])) + [
                str(self.blob.bounding_rect)[1:-1].replace(', ', sep),
                self.classification
            ])

    def __repr__(self) -> str:
        return f'{self._id}\tclass: {self.classification}\tps: {self._personScore}\tos: {self._objectScore}'

    def update(self, blob: 'Blob'):
        '''
        Evolve the entity's history with a new blob, and recompute it's speed.
        '''
        self._prev_blob = self._blob
        self._blob = blob

        self._speedX = self._blob.centroid[0] - self._prev_blob.centroid[0]
        self._speedY = self._blob.centroid[1] - self._prev_blob.centroid[1]

    def compute_scores(self, frame, diff, bg):
        '''
        Update the classification scores with the heuristic.
        '''

        if abs(self.speedX) > 1 or abs(self.speedY) > 1:
            self._personScore = min(self._personScore + 1, 40)
        else:
            self._personScore = max(self._personScore - 1, -40)

        if self._check_contour_similarity(frame, diff):
            self._objectScore = min(self._objectScore + 1, 40)
        else:
            self._objectScore = max(self._objectScore - 1, -40)

    def _check_contour_similarity(self,
                                  frame,
                                  diff,
                                  match_threshold=0.2) -> bool:
        '''
        Private function used to detect true or false object, by using the structural similarity index measure.
        Parameters:
        - frame: the original frame
        - diff: the frame after background subtraction
        Returns:
        - True if the detected similarity greater or equal to the provided threshold.
            
        '''
        x, y, w, h = self.blob.bounding_rect

        if (w < 7 or h < 7):
            return False

        frame_roi = frame[y:y + h, x:x + w, 0]
        diff_roi = diff[y:y + h, x:x + w]

        ssim_score = ssim(frame_roi,
                          diff_roi,
                          data_range=diff_roi.max() - diff_roi.min())

        return ssim_score >= match_threshold


class Blob:
    '''
    Represents a single blob, without making any temporal assumption.
    '''

    @cached_property
    def aspect(self):
        x, y, w, h = cv2.boundingRect(self._contour)
        return float(w) / h

    @cached_property
    def solidity(self):
        area = cv2.contourArea(self._contour)
        hull = cv2.convexHull(self._contour)
        hull_area = cv2.contourArea(hull)
        return float(area) / (hull_area + 0.00001)

    @cached_property
    def extent(self):
        area = cv2.contourArea(self._contour)
        _, _, w, h = cv2.boundingRect(self._contour)
        return float(area) / (w * h)

    @cached_property
    def moments(self):
        return cv2.moments(self._contour)

    @cached_property
    def centroid(self):
        return (self.moments['m10'] / (self.moments['m00'] + 0.00001),
                self.moments['m01'] / (self.moments['m00'] + 0.00001))

    @cached_property
    def perimeter(self):
        return cv2.arcLength(self._contour, True)

    @cached_property
    def area(self):
        return cv2.contourArea(self._contour)

    @cached_property
    def bounding_rect(self):
        return cv2.boundingRect(self._contour)

    @property
    def contour(self):
        return self._contour

    def corner(self, corner_code):
        '''
        Obtain the coordinates of a corner of the bounding rect.
        Parameters:
        - corner_code: Must be one of 'tl', 'tr', 'bl', 'br'  (t = top, b = bottom, l = left, r = right)
        '''
        x, y, h, w = self.bounding_rect

        if corner_code == 'br':
            return (x + w, y + h)
        elif corner_code == 'bl':
            return (x, y + h)
        elif corner_code == 'tl':
            return (x, y)
        elif corner_code == 'tr':
            return (x + w, y)
        else:
            raise ValueError('Expected one of (tl, tr, bl, br)')

    def intersection_area(self, other):
        '''
        Compute the intersection area between this blob's contour and another one.
        '''
        sx, sy = self.corner(corner_code='br')
        ox, oy = other.corner(corner_code='br')
        blank = np.zeros((max(sy, oy), max(sx, ox)))

        image1 = cv2.drawContours(blank.copy(), [self.contour],
                                  0,
                                  1,
                                  thickness=cv2.FILLED)

        image2 = cv2.drawContours(blank.copy(), [other.contour],
                                  0,
                                  1,
                                  thickness=cv2.FILLED)

        intersectionArea = cv2.countNonZero(cv2.bitwise_and(image1, image2))
        return intersectionArea

    # intersection_threshold = 15
    # perimeter_threshold = 0
    # aspect_threshold = 1.9
    # extent_threshold = 0.25
    # solidity_threshold = 0.2
    def similar(
        self,
        other: 'Blob',
        intersection_threshold: float = \
            hyperparams['tracking']['intersection_threshold'],
        aspect_threshold: float = \
            hyperparams['tracking']['aspect_threshold'],
        extent_threshold: float = \
            hyperparams['tracking']['extent_threshold'],
        solidity_threshold: float = \
            hyperparams['tracking']['solidity_threshold'],
    ):
        '''
        Check if the provided blob is similar to the current one.
        '''
        intersection = self.intersection_area(other)

        return all([
            intersection > intersection_threshold,
            abs(self.aspect - other.aspect) <= aspect_threshold,
            abs(self.extent - other.extent) <= extent_threshold,
            abs(self.solidity - other.solidity) <= solidity_threshold,
        ])

    def __str__(self) -> str:
        return f'Centroid: {self.centroid}\tArea: {self.area}'

    def __repr__(self) -> str:
        return f'Centroid: {self.centroid}\tArea: {self.area}'

    def __init__(self, contour):
        '''
        Parameters:
        - contour: the blob's contour
        '''
        self._contour = contour