import cv2
import os

# 1. 读取 MOT 输出结果
mot_result_path = "output/UCMCTrack_mot_output.txt"
results = {}  # 按帧号存储，每帧为列表，列表内每项为 (track_id, l, t, w, h)
with open(mot_result_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 假设文件内以逗号分隔
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        frame_id = int(parts[0])
        track_id = int(parts[1])
        l = int(float(parts[2]))
        t = int(float(parts[3]))
        w = int(float(parts[4]))
        h = int(float(parts[5]))
        if frame_id not in results:
            results[frame_id] = []
        results[frame_id].append((track_id, l, t, w, h))

# 2. 打开视频文件
video_path = "video/demo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件:", video_path)
    exit()

# 获取视频参数
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 3. 初始化视频写入对象（保存可视化结果）
output_video_path = "output/UCMCTrack_MOT.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_id = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 如果当前帧有检测结果，则在帧上绘制所有边界框和 track id
    if frame_id in results:
        for (track_id, l, t, w, h) in results[frame_id]:
            # 绘制矩形边界框（绿色，线宽2）
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
            # 在边界框上方绘制 track id
            cv2.putText(frame, f"ID:{track_id}", (l, max(t - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 将处理后的帧写入输出视频
    out.write(frame)
    frame_id += 1

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print("MOT 可视化视频已保存到:", output_video_path)
