import argparse
import os
import cv2
import numpy as np
import math

from deep_sort.deep_sort.deep.feature_extractor import Extractor


def read_mot_file(mot_file_path):
    """
    读取MOT输出文件，按track_id分组，并按frame_id排序每条轨迹中的检测。
    文件格式：frame_id, track_id, left, top, w, h, -1, -1, -1, -1
    """
    tracks = {}
    with open(mot_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append((frame_id, x, y, w, h))
    # 对每条轨迹按frame_id排序
    for tid in tracks:
        tracks[tid].sort(key=lambda det: det[0])
    return tracks


def crop_image(image, bbox, track_id=None, frame_id=None):
    """
    根据bbox从image中裁剪目标区域。
    bbox: (x, y, w, h)
    如果bbox超出图像范围，则裁剪掉超出部分；
    如果裁剪后图像大小为0，则打印该情况并返回None。
    """
    H, W = image.shape[:2]
    x, y, w, h = bbox
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    # 计算有效的裁剪区域
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        print(f"Track {track_id}, Frame {frame_id}: 裁剪后图像大小为0")
        return None
    crop_img = image[y1:y2, x1:x2]
    return crop_img


def compute_track_score(features):
    """
    计算一条轨迹的相似度得分：
    - 若轨迹只有1帧，得分为0；
    - 若轨迹帧数为N，则计算N-1个相邻帧特征的余弦相似度，
      然后取从高到低排序后前 ceil((N-1)/2) 个相似度的平均值作为得分。
    """
    num = features.shape[0]
    if num < 2:
        return 0.0
    sims = []
    for i in range(num - 1):
        f1 = features[i]
        f2 = features[i + 1]
        norm_product = np.linalg.norm(f1) * np.linalg.norm(f2)
        if norm_product == 0:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(f1, f2) / norm_product)
        sims.append(cos_sim)
    # 按相似度从高到低排序
    sims.sort(reverse=True)
    top_k = math.ceil(len(sims) / 2)
    top_half = sims[:top_k]
    score = sum(top_half) / len(top_half)
    return score


def parse_args():
    parser = argparse.ArgumentParser(description='get_seq')
    parser.add_argument("--seq", type=str, default='MOT17-13-SDP.txt')
    args = parser.parse_args()
    return args


def cal_score():
    # output_file_path = f"output/mot17/test/score.txt"
    # 文件路径设置
    # seq = "MOT17-13-SDP"
    # 命令行获取seq
    # args = parse_args()

    # print("seq_name", args.seq)

    image_dir = f"seq"
    # mot_file_path = f"score.txt"

    # 步骤1：读取MOT输出文件

    tracks_SORT = read_mot_file("output/SORT_mot_output.txt")
    tracks_DeepSORT = read_mot_file("output/DeepSORT_mot_output.txt")
    tracks_UCMCTrack = read_mot_file("output/UCMCTrack_mot_output.txt")
    # print(f"共读取到 {len(tracks)} 条轨迹")

    # 步骤4：初始化特征提取器
    # extractor = Extractor("D:\desk\Trajectory_Reconstruction_System\DeepSORT\deep_sort\deep_sort\deep\checkpoint\ckpt.t7", use_cuda=True)
    extractor = Extractor("deep_sort\deep_sort\deep\checkpoint\ckpt.t7", use_cuda=True)

    # ---------------------------------------------------------------------------------------------------
    # 存储每条轨迹的特征
    track_features = {}
    cnt = 0
    # 遍历每条轨迹，读取对应帧图像并裁剪目标区域

    for track_id, detections in tracks_SORT.items():
        cnt = cnt + 1
        # print(cnt)
        # 只考虑长度大于40的轨迹
        # print(len(detections))
        if len(detections) <= 40:
            continue

        # 对轨迹中的检测采样：排序之后每20帧取一帧
        sampled_detections = detections[::20]

        im_crops = []

        for (frame_id_, x, y, w, h) in sampled_detections:
            frame_id = frame_id_ - 1
            image_path = os.path.join(image_dir, f"{frame_id:06d}.jpg")
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            crop = crop_image(image, (x, y, w, h), track_id, frame_id)
            if crop is not None:
                im_crops.append(crop)
        if len(im_crops) == 0:
            print(f"轨迹 {track_id} 没有有效的目标裁剪区域")
            continue
        # 提取该轨迹所有裁剪目标的特征
        features = extractor(im_crops)  # features的形状为 (num_crops, 512)
        track_features[track_id] = features

    # 步骤5：计算每条轨迹的相似度得分
    track_scores = {}

    for track_id, features in track_features.items():
        score = compute_track_score(features)
        track_scores[track_id] = score

    # 步骤6：计算最终得分（所有轨迹得分求和后除以轨迹数）
    num_tracks = len(track_scores)
    if num_tracks == 0:
        final_score_SORT = float('inf')
    else:
        total_score = sum(track_scores.values())
        final_score_SORT = total_score / (len(tracks_SORT) ** (1 / 2))

    # ---------------------------------------------------------------------------------------------------
    # 存储每条轨迹的特征
    track_features = {}
    cnt = 0
    # 遍历每条轨迹，读取对应帧图像并裁剪目标区域

    for track_id, detections in tracks_DeepSORT.items():
        cnt = cnt + 1
        # print(cnt)
        # 只考虑长度大于40的轨迹
        # print(len(detections))
        if len(detections) <= 40:
            continue

        # 对轨迹中的检测采样：排序之后每20帧取一帧
        sampled_detections = detections[::20]

        im_crops = []

        for (frame_id_, x, y, w, h) in sampled_detections:
            frame_id=frame_id_-1
            image_path = os.path.join(image_dir, f"{frame_id:06d}.jpg")
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            crop = crop_image(image, (x, y, w, h), track_id, frame_id)
            if crop is not None:
                im_crops.append(crop)
        if len(im_crops) == 0:
            print(f"轨迹 {track_id} 没有有效的目标裁剪区域")
            continue
        # 提取该轨迹所有裁剪目标的特征
        features = extractor(im_crops)  # features的形状为 (num_crops, 512)
        track_features[track_id] = features

    # 步骤5：计算每条轨迹的相似度得分
    track_scores = {}

    for track_id, features in track_features.items():
        score = compute_track_score(features)
        track_scores[track_id] = score

    # 步骤6：计算最终得分（所有轨迹得分求和后除以轨迹数）
    num_tracks = len(track_scores)
    if num_tracks == 0:
        final_score_DeepSORT = float('inf')
    else:
        total_score = sum(track_scores.values())
        final_score_DeepSORT = total_score / (len(tracks_SORT) ** (1 / 2))

    # ---------------------------------------------------------------------------------------------------
    # 存储每条轨迹的特征
    track_features = {}
    cnt = 0
    # 遍历每条轨迹，读取对应帧图像并裁剪目标区域

    for track_id, detections in tracks_UCMCTrack.items():
        cnt = cnt + 1
        # print(cnt)
        # 只考虑长度大于40的轨迹
        # print(len(detections))
        if len(detections) <= 40:
            continue

        # 对轨迹中的检测采样：排序之后每20帧取一帧
        sampled_detections = detections[::20]

        im_crops = []

        for (frame_id_, x, y, w, h) in sampled_detections:
            frame_id=frame_id_-1
            image_path = os.path.join(image_dir, f"{frame_id:06d}.jpg")
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            crop = crop_image(image, (x, y, w, h), track_id, frame_id)
            if crop is not None:
                im_crops.append(crop)
        if len(im_crops) == 0:
            print(f"轨迹 {track_id} 没有有效的目标裁剪区域")
            continue
        # 提取该轨迹所有裁剪目标的特征
        features = extractor(im_crops)  # features的形状为 (num_crops, 512)
        track_features[track_id] = features

    # 步骤5：计算每条轨迹的相似度得分
    track_scores = {}

    for track_id, features in track_features.items():
        score = compute_track_score(features)
        track_scores[track_id] = score

    # 步骤6：计算最终得分（所有轨迹得分求和后除以轨迹数）
    num_tracks = len(track_scores)
    if num_tracks == 0:
        final_score_UCMCTrack = float('inf')
    else:
        total_score = sum(track_scores.values())
        final_score_UCMCTrack = total_score / (len(tracks_SORT) ** (1 / 2))

    return final_score_SORT, final_score_DeepSORT, final_score_UCMCTrack


if __name__ == "__main__":
    print(cal_score())

# 2025-03-09 08:00:00
