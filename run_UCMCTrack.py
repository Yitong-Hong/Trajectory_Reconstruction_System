
from tracker.ucmc import UCMCTrack
from tracker.kalman import TrackStatus
from eval.interpolation import interpolate
import os, time

import argparse


class Tracklet():
    def __init__(self, frame_id, box):
        self.is_active = False
        self.boxes = dict()
        self.boxes[frame_id] = box

    def add_box(self, frame_id, box):
        self.boxes[frame_id] = box

    def activate(self):
        self.is_active = True

seq="UCMCTrack_mot_output"
det_file="seq_det_res.txt"
cam_para="cam_para_file.txt"
wx=0.1
wy=0.1
vmax=0.5
a=10.0
cdt=30
high_score=0.6
conf_thresh=0.5
hp=True
cmc=False
u_ratio=0.05
v_ratio=0.05
add_cam_noise=True

# get w  h fps
def run_UCMCTrack(w,h,fps):
    seq_name = seq

    # eval_path="output"
    # orig_save_path = "ucmc_orig"  # ucmc_orig
    # if not os.path.exists(orig_save_path):
    #     os.makedirs(orig_save_path)
    # result_file="ucmc_orig/orig.txt"

    gmc_file=None
    gmc_path=None
    # if gmc_path is not None:
    #     gmc_file = os.path.join(gmc_path, f"GMC-{seq_name}.txt")

    # print(det_file)
    # print(cam_para)
    result_file="ucmc_orig/UCMCTrack_mot_output.txt"
    from detector.detector import Detector
    detector = Detector()
    detector.load(cam_para, w,h,det_file) # cam_para/MOT17/MOT17-13-SDP.txt
    print(f"seq_length = {detector.seq_length}")

    a1 = a
    a2 = a


    tracker = UCMCTrack(a1, a2, wx, wy, vmax, cdt, fps, high_score, cmc, detector)

    t1 = time.time()

    tracklets = dict()

    # result_file="output/UCMCTrack_mot_output.txt"
    with open(result_file, "w") as f:
        for frame_id in range(1, detector.seq_length + 1):
            print("frame_id", frame_id)
            dets = detector.get_dets(frame_id-1, conf_thresh)
            tracker.update(dets, frame_id)
            if hp:
                for i in tracker.tentative_idx:
                    t = tracker.trackers[i]
                    if (t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    if t.id not in tracklets:
                        tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                    else:
                        tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                for i in tracker.confirmed_idx:
                    t = tracker.trackers[i]
                    if (t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    if t.id not in tracklets:
                        tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                    else:
                        tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                    tracklets[t.id].activate()
            else:
                for i in tracker.confirmed_idx:
                    t = tracker.trackers[i]
                    if (t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    d = dets[t.detidx]
                    f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},"
                            f"{d.bb_height:.1f},{d.conf:.2f},-1,-1,-1\n")

        if hp:
            for frame_id in range(1, detector.seq_length + 1):
                for id in tracklets:
                    if tracklets[id].is_active:
                        if frame_id in tracklets[id].boxes:
                            box = tracklets[id].boxes[frame_id]
                            f.write(
                                f"{frame_id},{id},{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},-1,-1,-1,-1\n")

    interpolate("ucmc_orig", "output", n_min=3, n_dti=cdt, is_enable=True)
    # 差值之前结果在result_file（output/mot17/val/MOT17-02/MOT17-13-SDP.txt
    # 差值之后的结果在eval_path（output/mot17/val）里面，eval用这个

    # print(f"Time cost: {time.time() - t1:.2f}s")

# run_UCMCTrack(1360,764,20)