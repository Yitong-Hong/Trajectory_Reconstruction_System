"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):  # 采用两种算法，时间复杂度均为O(n^3)
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        # 优先采用Jonker-Volgenant 算法，通常更高效
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)  # 匈牙利算法

    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)  # (1,8,5)
    bb_test = np.expand_dims(bb_test, 1)  # (7,1,5)

    # 广播机制
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])  # (7, 8)
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])  # (7, 8)
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])  # (7, 8)
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])  # (7, 8)
    w = np.maximum(0., xx2 - xx1)  # (7, 8)
    h = np.maximum(0., yy2 - yy1)  # (7, 8)
    wh = w * h  # (7, 8)
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)  # (7, 8)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(  # 状态转移
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(  # 状态到观测的映射
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.  # 观测噪声
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.  # 状态预测误差矩阵
        self.kf.Q[-1, -1] *= 0.01  # 过程噪声
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0  # 之前连串了多少，所以初始化为0。因为每轮是先predict,考察连串多少，所以不应该包含本轮是否匹配到
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0  # 重置time_since_update
        self.history = []  # 清空历史，这个历史指的是tracker保存的距离上次update的box
        self.hits += 1
        self.hit_streak += 1
        # 一连串的命中+1，从初始化之后，每次update都加，一旦哪次没加，predict时候赋为0
        self.kf.update(convert_bbox_to_z(bbox))

    # 维护卡尔曼滤波器相关的，实现更新预测，维护time_since_update,hit_streak为后面删除做准备
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
            # 马上要预测了，以防面积更新之后为非正数，
            # 如果可能变为非正数，那就把面积的变化速度变为0
        self.kf.predict()  # 调用api实现KF预测
        self.age += 1  # 此算法中没用到age做什么事，我认为如果用到age,应该在update中更新
        if (self.time_since_update > 0):  # 上轮没update，到本轮，streak断了(正常先predict在update)
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # history[-1].shape (1,4)
        return self.history[-1]  # 返回history中最新的，即当下状态

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)  # 得到最新的状态（更新到什么时候就是什么时候）


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):  # 如果此时没有tracker,返回 matches, np.array(unmatched_detections), np.array(unmatched_trackers)
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)), np.empty((0, 5), dtype=int))

    iou_matrix = iou_batch(detections, trackers)  # 得到detection trackers的关系矩阵
    # (num_detections, num_trackers)

    if min(iou_matrix.shape) > 0:  # 非空矩阵
        a = (iou_matrix > iou_threshold).astype(np.int32)  # 过滤掉小于阈值的iou (边)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:  # 如果过滤之后就不存在冲突，则直接作为答案
            matched_indices = np.stack(np.where(a), axis=1)  # 得到(detection_id,tracker_id)的二元组集合
        else:
            matched_indices = linear_assignment(-iou_matrix)  # 用-iou_matrix匹配
    else:  # 此时说明没有detection
        matched_indices = np.empty(shape=(0, 2))  # 没有匹配对
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)
    # 把没匹配detections，trackers装入

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        # 合并前 matches = [array([[0, 1]]), array([[2, 3]]), array([[4, 5]])]
        # 合并后
        # array([
        #     [0, 1],
        #     [2, 3],
        #     [4, 5]
        # ])

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age  # 最大年龄，常数
        self.min_hits = min_hits  # 实习期+1
        self.iou_threshold = iou_threshold  # iou 门限
        self.trackers = []  # tracker列表
        self.frame_count = 0  # 第几帧

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # print()
        # print("frame_count", self.frame_count)
        # get predicted locations from existing trackers.
        # len(trackers) 追踪器数量，假设现在有2个
        trks = np.zeros((len(self.trackers), 5))
        # 预测位置的二维数组，每行表示一个边界框和一个占位符 2*5
        to_del = []  # 需要删除的追踪器索引
        ret = []  # 最终的追踪结果

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]  # (1,4)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = (
            associate_detections_to_trackers(dets, trks, self.iou_threshold))

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            # print("add tracker", i)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 正序删除+遍历会出问题，删除某位，后面前移，下一位变成下下位了
            d = trk.get_state()[0]
            # 此轮得有，否则免谈
            if (trk.time_since_update < 1) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                # +1 as MOT benchmark requires positive
                # 满足上述条件，才返回这个追踪结果。
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                # print("delete tracker", i)
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))


max_age = 1
display = False
min_hits = 3
iou_threshold = 0.3


def run_SORT():

    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display

    if not os.path.exists('output'):
        os.makedirs('output')

    mot_tracker = Sort(max_age=max_age,
                       min_hits=min_hits,
                       iou_threshold=iou_threshold)  # create instance of the SORT tracker
    seq = "SORT_mot_output"
    seq_dets_fn = "seq_det_res.txt"
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')

    with open(os.path.join('output', '%s.txt' % seq), 'w') as out_file:
        # print("Processing %s." % seq)
        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1

            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            for d in trackers:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                      file=out_file)

    # print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
    #     total_time, total_frames, total_frames / total_time))

