from collections import OrderedDict, Counter
from typing import Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist

import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
import detectron2.utils.video_visualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

import pickle

BAG_LABEL = 1
PERSON_LABEL = 0

SAVE_PREDICTIONS = False
SAVED_PREDICTIONS = []

def compute_center(bounding_boxes):
    # type: (np.ndarray) -> np.ndarray

    x_dist = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    y_dist = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    centers = bounding_boxes[:, 0:2] + 0.5 * np.stack((x_dist, y_dist), axis=1)
    centers_3d = add_z_coordinate(centers)
    return centers_3d


def add_z_coordinate(centers):
    # type: (np.ndarray) -> np.ndarray

    return np.concatenate((centers, np.zeros(shape=(centers.shape[0], 1))), axis=1)


def split_bag_persons(centers, labels):
    # type: (np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray]

    assert isinstance(labels, np.ndarray)
    assert isinstance(centers, np.ndarray)
    print(labels)
    bag_bounding_centers = centers[labels == BAG_LABEL]
    persons_bounding_centers = centers[labels == PERSON_LABEL]

    return bag_bounding_centers, persons_bounding_centers


# def extract_bag_to_ppl_vectors(boudning_boxes, labels):
#     centers = compute_center(boudning_boxes)
#     bag_centers, persons_centers = split_bag_persons(centers, labels)
#     distances = cdist(bag_centers, persons_centers)
#     ind = distances.argmin(axis=1)
#     dist = distances[np.arange(len(ind)), ind]
#
#     return dist, ind


class SimpleBagTracker:

    def __init__(self, self_association_thres, bag_person_thres):
        # type: (float, float) -> None


        self.all_centers = {'bags': OrderedDict(),
                            'persons': OrderedDict()}  # Stores seen bounding box centers by object id
        self.prev_frame_ids = {'bags': [],
                               'persons': []}  # Stores ids of objects observed in the last frame

        self.bag_person_association = dict()  # Maps bag id to bag owner id
        self.bag_person_dist = dict()  # Maps bag id to distance to the bag owner
        self.instance_count = {'bags': 0,
                               'persons': 0}  # Counts how many bags and persons have been seen
        self.bag_person_thres = bag_person_thres
        self.self_association_thres = self_association_thres
        self.prev_frame_kept = {'bags': False,
                                'persons': False}  # Tracks if last frame's bounding boxes were kept or ignored
        self.keep_frame_counter = {'bags': 0,
                                   'persons': 0}  # Counts how many frames back object centers have been stored

    def update(self, boxes, labels):
        # type: (np.ndarray, np.ndarray) -> None

        centers = compute_center(boxes)
        bag_bounding_centers, persons_bounding_centers = split_bag_persons(centers, labels)

        self.frame2frame_association(persons_bounding_centers, 'persons')
        self.frame2frame_association(bag_bounding_centers, 'bags')
        self.update_bag_person_association()

        print(self.prev_frame_ids)

    def is_unattended(self, bag_id):
        # type: (int) -> bool

        person_id = self.bag_person_association[bag_id]
        if person_id is None:
            return True
        person_center = self.all_centers['persons'][person_id]
        bag_center = self.all_centers['bags'][bag_id]

        if np.sqrt(((person_center - bag_center) ** 2).sum()) > self.bag_person_thres:
            return True

        return False

    def frame2frame_association(self, new_centers, tag):
        # type: (np.ndarray, str) -> None

        frame_ids = []
        frame_centers = []
        new_frame_unused_centers = list(range(new_centers.shape[0]))
        if len(self.prev_frame_ids[tag]) > 0 and len(new_centers) > 0:
            prev_frame_centers = np.stack([self.all_centers[tag][id] for id in self.prev_frame_ids[tag]], axis=0)
            distances = cdist(prev_frame_centers, new_centers)

            cc_in_new_frame_index = distances.argmin(axis=1)
            new_frame_unused_centers = list(set(new_frame_unused_centers) - set(cc_in_new_frame_index.tolist()))

            min_dist = distances[range(len(self.prev_frame_ids[tag])), cc_in_new_frame_index]
            index_counter = Counter(cc_in_new_frame_index)

            for dist, prev_frame_id, new_center, index in zip(min_dist,
                                                              self.prev_frame_ids[tag],
                                                              new_centers[cc_in_new_frame_index],
                                                              cc_in_new_frame_index):

                if dist < self.self_association_thres and index_counter[index] <= 1:
                    # case where there is a unique closest center
                    self.all_centers[tag][prev_frame_id] = new_center
                    frame_ids.append(prev_frame_id)
                    frame_centers.append(new_center)
                elif dist > self.self_association_thres and index_counter[index] <= 1:
                    # case where the closest frame is too far away
                    self.all_centers[tag][self.instance_count[tag]] = new_center
                    frame_ids.append(self.instance_count[tag])
                    frame_centers.append(new_center)
                    self.instance_count[tag] += 1
                else:
                    # case where one new center is closest to several centers
                    other_dists = min_dist[cc_in_new_frame_index == index]
                    if dist <= other_dists.min():
                        self.all_centers[tag][prev_frame_id] = new_center
                        frame_ids.append(prev_frame_id)
                        frame_centers.append(new_center)

        # add the new centers which were not closest to any old center
        for new_center in new_centers[new_frame_unused_centers, :]:
            self.all_centers[tag][self.instance_count[tag]] = new_center
            frame_ids.append(self.instance_count[tag])
            frame_centers.append(new_center)
            self.instance_count[tag] += 1

        if frame_ids:
            self.prev_frame_ids[tag] = frame_ids
            self.prev_frame_kept[tag] = False
            self.keep_frame_counter[tag] = 0
        else:
            self.keep_frame_counter[tag] += 1
            if self.keep_frame_counter[tag] > 8:
                for id in self.prev_frame_ids[tag]:
                    self.all_centers[tag][id] = np.array([np.Inf, np.Inf, np.Inf])
                self.prev_frame_ids[tag] = []
                self.prev_frame_kept[tag] = False
            else:
                self.prev_frame_kept[tag] = True

        # print(frame_ids, self.prev_frame_ids[tag])
        # print(self.all_centers[tag])

    def update_bag_person_association(self):
        # type: () -> None
        """
        Iterates over all detected bags in the last frame (current frame) and updates the bag-person association and
        the bag person distance.
        """

        for bag_id in self.prev_frame_ids['bags']:
            if bag_id not in self.bag_person_association or self.bag_person_association[bag_id] is None:
                # Case were the bag has not previous owner
                person_id, dist = self.find_closest_person_to_bag(bag_id)
                self.bag_person_association[bag_id] = person_id
                self.bag_person_dist[bag_id] = dist
            elif self.bag_person_association[bag_id] not in self.prev_frame_ids['persons']:
                # Case were the bags owner as not observed in the current frame
                self.bag_person_dist[bag_id] = float('inf')
            else:
                # Case were both bag and owner were observed in the current frame
                bag_person_vector = (self.all_centers['persons'][self.bag_person_association[bag_id]] -
                                     self.all_centers['bags'][bag_id])
                self.bag_person_dist[bag_id] = np.sqrt(np.power(bag_person_vector, 2).sum())

    def find_closest_person_to_bag(self, bag_id):
        # type: (int) -> Tuple[Optional[int], float]
        """
        Checks for closest person in the current frame given an id of a detected bag.
        Returns the id of the person and the distance given that a person could be found with a distance below the
        bag_person_thres threshold.

        Args:
            bag_id: Id of a bag observed in the current frame.
        Returns:
            person_id: Id of the closest person or None if no person could be found with a distance smaller than
                bag_person_thres
            distance: Distance in pixels between the person and the bag. Inf if not person could be found.
        """
        bag_center = self.all_centers['bags'][bag_id]
        dists = []
        for person_id in self.prev_frame_ids['persons']:
            person_center = self.all_centers['persons'][person_id]
            dists.append(np.sqrt(np.power(person_center - bag_center, 2).sum()))
        if not self.prev_frame_ids['persons']:
            return None, float('inf')
        closest_person_ind = int(np.array(dists).argmin())
        if dists[closest_person_ind] < self.bag_person_thres:
            return self.prev_frame_ids['persons'][closest_person_ind], dists[closest_person_ind]
        else:
            return None, float('inf')



def draw_instance_predictions(visualizer, frame, predictions, tracker):

    frame_visualizer = Visualizer(frame, visualizer.metadata)
    num_instances = len(predictions)
    if num_instances == 0:
        return frame_visualizer.output

    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if predictions.has("pred_masks"):
        masks = predictions.pred_masks
        # mask IOU is not yet enabled
        # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
        # assert len(masks_rles) == num_instances
    else:
        masks = None

    detected = [
        detectron2.utils.video_visualizer._DetectedInstance(classes[i], boxes[i], mask_rle=None, color=None, ttl=8)
        for i in range(num_instances)
    ]
    colors = visualizer._assign_colors(detected)

    labels = detectron2.utils.video_visualizer._create_text_labels(classes, scores,
                                                                   visualizer.metadata.get("thing_classes", None))

    if visualizer._instance_mode == ColorMode.IMAGE_BW:
        # any() returns uint8 tensor
        frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
            (masks.any(dim=0) > 0).numpy() if masks is not None else None
        )
        alpha = 0.3
    else:
        alpha = 0.5

    frame_visualizer.overlay_instances(
        boxes=None if masks is not None else boxes,  # boxes are a bit distracting
        masks=masks,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )

    for bag_id in tracker.prev_frame_ids['bags']:
        bag_center = tracker.all_centers['bags'][bag_id]
        if bag_id in tracker.bag_person_association:
            person_id = tracker.bag_person_association[bag_id]
            if person_id is not None and person_id in tracker.prev_frame_ids['persons']:
                person_center = tracker.all_centers['persons'][person_id]

                if tracker.is_unattended(bag_id):
                    frame_visualizer.draw_line(
                        [bag_center[0], person_center[0]],
                        [bag_center[1], person_center[1]],
                        'r'
                    )
                else:
                    frame_visualizer.draw_line(
                        [bag_center[0], person_center[0]],
                        [bag_center[1], person_center[1]],
                        'g'
                    )

        if tracker.is_unattended(bag_id):
            frame_visualizer.draw_text(
                'abandoned',
                tuple(bag_center[0:2]),
                color='r'
            )

    return frame_visualizer.output


class AbandonmentDetector(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.tracker = SimpleBagTracker(150, 200)
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):

        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):

        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, tracker):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                tracker.update(boxes=predictions.pred_boxes.tensor.numpy(), labels=predictions.pred_classes.numpy())

                if SAVE_PREDICTIONS:
                    SAVED_PREDICTIONS.append(predictions)
                    if len(SAVED_PREDICTIONS) == 100:
                        with open('predictions.pkl', 'wb') as fp:
                            pickle.dump(SAVED_PREDICTIONS, fp)
                            print('Saving done!')

                vis_frame = draw_instance_predictions(video_visualizer, frame, predictions, tracker)

            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)

        for frame in frame_gen:
            yield process_predictions(frame, self.predictor(frame), self.tracker)

