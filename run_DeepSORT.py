

import cv2
import numpy as np
import os

from AIDetector_pytorch import Detector

def run_DeepSORT():
    # args = parse_args()
    # phase = args.phase
    # pattern = os.path.join(args.seq_path, phase, '*')
    total_frames = 0

    det = Detector()

    seq="DeepSORT_mot_output"
    # print("有啊",os.path.exists("D:\desk\Trajectory_Reconstruction_System\seq_det_res.txt"))
    seq_dets_fn = "seq_det_res.txt"
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    # print(seq_dets.shape)
    # print(seq)
    # 由于数据集的格式是[frame_id,track_id,l,t,w,h,confidence],所以需要转换成[frame_id,track_id,x1,y1,x2,y2,confidence]
    seq_without_txt=os.path.basename(seq_dets_fn).replace('.txt', '')
    img_folder = "seq"

    # with open(os.path.join('output', 'MOT17','test','%s' % (seq)), 'w') as out_file:
    with open(os.path.join('output', '%s.txt' % seq), 'w') as out_file:
        # print("Processing %s." % (seq))
        for f in range(int(seq_dets[:, 0].max())):
            frame=f+1
            print("frame ",frame)
            dets = seq_dets[seq_dets[:, 0] == frame, 2:8]
            # print("dets[0] [x1,y1,w,h]",dets[0])
            dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            # print("dets[0] [x1,y1,x2,y2]", dets[0])
            img_path = os.path.join(img_folder, f"{frame-1:06d}.jpg")
            # print(img_path)
            im = cv2.imread(img_path)  # im represents the image
            # Make sure the image was loaded properly

            if im is None:
                print(f"Error loading image {img_path}")
                continue


            result = det.feedCap(im,dets)# list,each element: (x1, y1, x2, y2, track_id)
            for d in result:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1' % (frame, d[5], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                      file=out_file)
                # print('%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1' % (frame, d[5], d[0], d[1], d[2] - d[0], d[3] - d[1]))


