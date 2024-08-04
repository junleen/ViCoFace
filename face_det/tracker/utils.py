#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import cv2
import numpy as np

def extract_opencv_generator(filename):
    """extract_opencv_generator.

    :param filename: str, the filename for video.
    """
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


def get_landmarks(multi_sub_landmarks):
    """get_landmarks.

    :param multi_sub_landmarks: dict, a dictionary contains landmarks, bbox, and landmarks_scores.
    """

    landmarks = [None] * len( multi_sub_landmarks["landmarks"])
    for frame_idx in range(len(landmarks)):
        if len(multi_sub_landmarks["landmarks"][frame_idx]) == 0:
            continue
        else:
            # -- decide person id using maximal bounding box  0: Left, 1: top, 2: right, 3: bottom, probability
            max_bbox_person_id = 0
            max_bbox_len = multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][2] + \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][3] - \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][0] - \
                           multi_sub_landmarks["bbox"][frame_idx][max_bbox_person_id][1]
            landmark_scores = multi_sub_landmarks["landmarks_scores"][frame_idx][max_bbox_person_id]
            for temp_person_id in range(1, len(multi_sub_landmarks["bbox"][frame_idx])):
                temp_bbox_len = multi_sub_landmarks["bbox"][frame_idx][temp_person_id][2] + \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][3] - \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][0] - \
                                multi_sub_landmarks["bbox"][frame_idx][temp_person_id][1]
                if temp_bbox_len > max_bbox_len:
                    max_bbox_person_id = temp_person_id
                    max_bbox_len = temp_bbox_len
                    landmark_scores = multi_sub_landmarks['landmarks_scores'][frame_idx][temp_person_id]
            if landmark_scores[17:].min() >= 0.2:
                landmarks[frame_idx] = multi_sub_landmarks["landmarks"][frame_idx][max_bbox_person_id]
    return landmarks

def get_standard_landmarks(resolution):
    array = np.array([
        [0.17857143, 0.33928571],
        [0.1875    , 0.41964286],
        [0.19642857, 0.50446429],
        [0.20535714, 0.58482143],
        [0.23214286, 0.65625   ],
        [0.27232143, 0.71875   ],
        [0.33928571, 0.76785714],
        [0.41964286, 0.79464286],
        [0.5       , 0.80357143],
        [0.58035714, 0.79464286],
        [0.66517857, 0.76785714],
        [0.72767857, 0.71875   ],
        [0.76785714, 0.65178571],
        [0.79017857, 0.58035714],
        [0.80357143, 0.5       ],
        [0.8125    , 0.41517857],
        [0.82142857, 0.33482143],
        [0.22767857, 0.29017857],
        [0.26785714, 0.21428571],
        [0.33482143, 0.20089286],
        [0.40178571, 0.20535714],
        [0.45982143, 0.22767857],
        [0.54910714, 0.22767857],
        [0.60714286, 0.20535714],
        [0.67410714, 0.19642857],
        [0.73660714, 0.21428571],
        [0.77678571, 0.29017857],
        [0.50446429, 0.29017857],
        [0.50446429, 0.34375   ],
        [0.50446429, 0.38392857],
        [0.50446429, 0.42857143],
        [0.44642857, 0.49107143],
        [0.47321429, 0.5       ],
        [0.5       , 0.50892857],
        [0.52678571, 0.5       ],
        [0.55357143, 0.49107143],
        [0.29910714, 0.31696429],
        [0.33928571, 0.29017857],
        [0.38392857, 0.29017857],
        [0.41964286, 0.31696429],
        [0.38392857, 0.33035714],
        [0.34375   , 0.33482143],
        [0.58482143, 0.31696429],
        [0.62053571, 0.29017857],
        [0.66517857, 0.29017857],
        [0.70535714, 0.31696429],
        [0.66071429, 0.33482143],
        [0.62053571, 0.33035714],
        [0.39285714, 0.60267857],
        [0.42857143, 0.58482143],
        [0.47321429, 0.57589286],
        [0.5       , 0.58035714],
        [0.53125   , 0.57589286],
        [0.57142857, 0.58482143],
        [0.60714286, 0.60267857],
        [0.57589286, 0.61607143],
        [0.53125   , 0.62053571],
        [0.5       , 0.62053571],
        [0.46875   , 0.62053571],
        [0.42410714, 0.61607143],
        [0.39732143, 0.60267857],
        [0.47321429, 0.60267857],
        [0.5       , 0.60267857],
        [0.53125   , 0.60714286],
        [0.60714286, 0.60267857],
        [0.53125   , 0.58928571],
        [0.5       , 0.58928571],
        [0.46875   , 0.58928571]])
    
    if isinstance(resolution, int):
        HW = np.array([resolution, resolution])
    else:
        h, w = resolution[0], resolution[1]
        HW = np.array([w, h])
    return array*HW
    