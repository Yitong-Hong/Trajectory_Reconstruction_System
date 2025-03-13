import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import torch


class TRSApp:
    def __init__(self, master):
        self.master = master
        master.title("轨迹还原系统 (TRS)")

        # 初始化变量
        self.video_path = None
        self.image_path = None
        self.start_time = None
        self.video_fps = None
        self.img_width = None
        self.img_height = None

        # 上传视频按钮
        self.upload_video_btn = tk.Button(master, text="上传视频", command=self.upload_video)
        self.upload_video_btn.pack(pady=5)

        # 上传图像按钮
        self.upload_image_btn = tk.Button(master, text="上传图像", command=self.upload_image)
        self.upload_image_btn.pack(pady=5)

        # 视频起始时间输入
        self.start_time_label = tk.Label(master, text="视频起始时间 (例如：2025-03-09 08:00:00):")
        self.start_time_label.pack(pady=5)
        self.start_time_entry = tk.Entry(master, width=30)
        self.start_time_entry.pack(pady=5)

        # 处理按钮
        self.process_btn = tk.Button(master, text="开始处理", command=self.process_files)
        self.process_btn.pack(pady=10)

        # 可视化检测结果按钮
        self.visualize_btn = tk.Button(master, text="可视化检测结果", command=self.test_det)
        self.visualize_btn.pack(pady=5)

        # 信息显示框
        self.info_text = tk.Text(master, height=15, width=70)
        self.info_text.pack(pady=5)

    def upload_video(self):
        # 弹出文件选择对话框，选择视频文件（支持mp4、avi、mov等格式）
        self.video_path = filedialog.askopenfilename(title="选择视频文件",
                                                     filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.info_text.insert(tk.END, f"已选择视频文件：{self.video_path}\n")

    def upload_image(self):
        # 弹出文件选择对话框，选择图像文件（支持jpg、png、bmp等格式）
        self.image_path = filedialog.askopenfilename(title="选择图像文件",
                                                     filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")])
        if self.image_path:
            self.info_text.insert(tk.END, f"已选择图像文件：{self.image_path}\n")

    def process_files(self):
        # 获取视频起始时间
        self.start_time = self.start_time_entry.get().strip()
        if not self.video_path or not self.image_path or not self.start_time:
            messagebox.showerror("错误", "请确保视频、图像文件以及视频起始时间都已提供。")
            return

        # 打开视频文件，获取 FPS
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件。")
            return
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.info_text.insert(tk.END, f"视频 FPS：{self.video_fps}\n")

        # 读取图像，获取图像尺寸
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("错误", "无法读取图像文件。")
            return
        self.img_height, self.img_width = image.shape[:2]
        self.info_text.insert(tk.END, f"图像尺寸：宽 {self.img_width} 像素, 高 {self.img_height} 像素\n")

        # 创建保存帧图像的文件夹 "seq"
        seq_folder = "seq"
        if not os.path.exists(seq_folder):
            os.makedirs(seq_folder)

        # 将视频分解为帧，并保存到 "seq" 文件夹下
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(seq_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        cap.release()

        self.info_text.insert(tk.END, f"视频已分解为 {frame_count} 帧，保存在文件夹 '{seq_folder}' 中。\n")
        self.info_text.insert(tk.END, f"视频起始时间：{self.start_time}\n")

        # ------------------------- 添加检测功能 -------------------------
        self.info_text.insert(tk.END, "正在加载检测模型，请稍候...\n")
        try:
            # 加载 YOLOv5s 模型（首次运行时会自动下载模型权重）
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        except Exception as e:
            messagebox.showerror("错误", f"加载检测模型失败: {str(e)}")
            return
        self.info_text.insert(tk.END, "检测模型加载完成！\n")

        # 可根据需要调整置信度阈值
        model.conf = 0.5

        # 打开输出文件，写入检测结果
        output_file = "seq_det_res.txt"
        with open(output_file, "w") as f_out:
            # 获取 seq 文件夹下所有图像，按文件名排序（假定文件名格式为 frame_000000.jpg）
            frame_files = sorted(
                [file for file in os.listdir(seq_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for frame_file in frame_files:
                frame_path = os.path.join(seq_folder, frame_file)
                # 提取帧ID，例如 "frame_000001.jpg" 提取 000001 转为整数并加 1
                try:
                    frame_id_str = frame_file.split('_')[1].split('.')[0]
                    frame_id = int(frame_id_str) + 1
                except Exception as e:
                    self.info_text.insert(tk.END, f"无法解析文件名 {frame_file} 的帧ID，跳过该文件。\n")
                    continue

                # 使用模型检测当前帧
                results = model(frame_path)
                # 将检测结果转换为 DataFrame，包含 xmin, ymin, xmax, ymax, confidence 等信息
                detections = results.pandas().xyxy[0]
                det_id = 1  # 检测目标编号从1开始
                for index, row in detections.iterrows():
                    xmin = row['xmin']
                    ymin = row['ymin']
                    xmax = row['xmax']
                    ymax = row['ymax']
                    conf = row['confidence']
                    # 计算宽度和高度
                    width = xmax - xmin
                    height = ymax - ymin
                    # 构造输出行：{frame_id,det_id,l,t,w,h,conf,-1,-1,-1}
                    line = f"{frame_id},{det_id},{xmin:.1f},{ymin:.1f},{width:.1f},{height:.1f},{conf:.2f},-1,-1,-1\n"
                    f_out.write(line)
                    det_id += 1
                self.info_text.insert(tk.END, f"帧 {frame_id} 检测完成，共检测到 {det_id - 1} 个目标。\n")
        self.info_text.insert(tk.END, f"所有帧检测完成，结果已保存到 {output_file}\n")

    def test_det(self):
        """
        读取检测结果文件 seq_det_res.txt，并对 seq 文件夹下每一帧图像绘制检测框进行可视化。
        按下 ESC 键可退出可视化。
        """
        output_file = "seq_det_res.txt"
        seq_folder = "seq"
        if not os.path.exists(output_file):
            messagebox.showerror("错误", f"检测结果文件 {output_file} 不存在，请先进行检测。")
            return

        # 读取检测结果，并按帧号分组
        det_dict = {}
        with open(output_file, "r") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 7:
                    continue
                try:
                    frame_id = int(parts[0])
                    det_id = int(parts[1])
                    l = float(parts[2])
                    t = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6])
                except Exception as e:
                    self.info_text.insert(tk.END, f"解析检测结果失败: {line}\n")
                    continue
                if frame_id not in det_dict:
                    det_dict[frame_id] = []
                det_dict[frame_id].append((det_id, l, t, w, h, conf))

        # 按帧号排序并遍历每一帧
        for frame_id in sorted(det_dict.keys()):
            # 根据 frame_id 构造图像文件名：frame_id 从1开始，文件名格式为 frame_{frame_id-1:06d}.jpg
            frame_filename = os.path.join(seq_folder, f"frame_{frame_id - 1:06d}.jpg")
            if not os.path.exists(frame_filename):
                self.info_text.insert(tk.END, f"文件 {frame_filename} 不存在，跳过该帧。\n")
                continue
            img = cv2.imread(frame_filename)
            if img is None:
                self.info_text.insert(tk.END, f"无法读取文件 {frame_filename}，跳过该帧。\n")
                continue

            # 绘制每个检测框
            for det in det_dict[frame_id]:
                det_id, l, t, w, h, conf = det
                pt1 = (int(l), int(t))
                pt2 = (int(l + w), int(t + h))
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
                label = f"{det_id}:{conf:.2f}"
                cv2.putText(img, label, (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("检测结果", img)
            key = cv2.waitKey(0)  # 按任意键显示下一帧，ESC 键退出
            if key == 27:  # ESC键退出
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()  # 创建根窗口
    app = TRSApp(root)
    root.mainloop()
    # 如果不使用 GUI，也可以直接调用 test_det 进行检测结果可视化：
    app.test_det()
