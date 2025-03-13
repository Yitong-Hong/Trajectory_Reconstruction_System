import tkinter as tk
from tkinter import filedialog, messagebox
from score import cal_score
# ★★ 新增导入 YOLOv8 ★★
from ultralytics import YOLO
import shutil
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid
from torchvision import transforms
from run_SORT import run_SORT
from run_UCMCTrack import run_UCMCTrack
from run_DeepSORT import run_DeepSORT
from PIL import Image
from datetime import datetime, timedelta


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_folders_and_files():
    # 清空指定的文件夹内容
    folders_to_clear = ['seq', 'video', 'appearance', 'appearance_candidate', 'photo']
    for folder in folders_to_clear:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"清空文件夹 {folder} 时出错：{str(e)}")

    # 清空 txt 文件内容
    txt_folders = ['output', 'ucmc_orig']
    for folder in txt_folders:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder, filename)
                    try:
                        with open(file_path, 'w') as file:
                            file.truncate(0)  # 清空文件内容
                    except Exception as e:
                        print(f"清空文件 {file_path} 内容时出错：{str(e)}")


def parse_mot_result(mot_result_path):
    """
    解析 MOT 结果文件，按 track_id 分组，返回字典：{ track_id: [(frame_id, l, t, w, h), ...], ... }
    """
    track_detections = {}
    with open(mot_result_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            l = float(parts[2])
            t = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            if track_id not in track_detections:
                track_detections[track_id] = []
            track_detections[track_id].append((frame_id, l, t, w, h))
    # 对每个轨迹按照 frame_id 排序
    for tid in track_detections:
        track_detections[tid].sort(key=lambda x: x[0])
    return track_detections


def select_candidates(detections, num_candidates=5):
    """
    给定一个轨迹的所有检测，返回候选检测列表。
    如果检测数大于 num_candidates，则均匀采样 num_candidates 个；否则返回全部。
    """
    N = len(detections)
    if N <= num_candidates:
        return detections
    else:
        indices = np.linspace(0, N - 1, num_candidates, dtype=int)
        return [detections[i] for i in indices]


def save_candidate_images(track_detections, seq_folder, candidate_root):
    """
    对每个轨迹，从其检测中选取候选检测，
    根据 frame_id 在 seq 文件夹中加载图像，裁剪 bbox 区域，
    保存到 candidate_root/track_id/ 下。
    """
    if not os.path.exists(candidate_root):
        os.makedirs(candidate_root)

    for track_id, detections in track_detections.items():
        candidates = select_candidates(detections, num_candidates=5)
        track_folder = os.path.join(candidate_root, str(track_id))
        if not os.path.exists(track_folder):
            os.makedirs(track_folder)
        for i, (frame_id, l, t, w, h) in enumerate(candidates):
            # 根据 frame_id 生成图像文件名，注意 frame_id 从 1 开始，文件命名从 000000.jpg 开始
            img_filename = os.path.join(seq_folder, f"{frame_id - 1:06d}.jpg")
            if not os.path.exists(img_filename):
                print(f"图像文件 {img_filename} 不存在，跳过 track {track_id} 的候选 {i + 1}.")
                continue
            img = cv2.imread(img_filename)
            if img is None:
                print(f"读取图像 {img_filename} 失败，跳过 track {track_id} 的候选 {i + 1}.")
                continue
            # 裁剪检测框区域，注意确保不超过图像边界
            x1 = int(round(l))
            y1 = int(round(t))
            x2 = int(round(l + w))
            y2 = int(round(t + h))
            h_img, w_img = img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            cropped_img = img[y1:y2, x1:x2]
            candidate_filename = os.path.join(track_folder, f"candidate_{i + 1}.jpg")
            cv2.imwrite(candidate_filename, cropped_img)
            print(f"保存 track {track_id} 的候选图像到 {candidate_filename}")


def calculate_time(start_time, frame_id, fps):
    """
    计算视频的时间，基于起始时间和帧率
    """
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    frame_time_offset = frame_id / fps  # 每帧的时间偏移（秒）
    frame_time = start_dt + timedelta(seconds=frame_time_offset)
    return frame_time.strftime("%H:%M:%S")  # 只显示时:分:秒

def extract_feature(image_path, model, transform, device):
    """
    加载 image_path 对应的图像，预处理后用 model 提取特征，返回归一化后的特征向量（tensor）
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img_tensor)
        feat = feat.view(feat.size(0), -1)
    feat = F.normalize(feat, p=2, dim=1)
    return feat

def re_id(best_res):
    # 路径设置
    mot_result_path = best_res  # MOT 结果文件
    seq_folder = "seq"  # 图像序列文件夹
    candidate_root = "appearance_candidate"  # 存放候选图像的文件夹
    final_appearance_folder = "appearance"  # 存放最终筛选图像的文件夹
    target_img_path = os.path.join("photo", "target.jpg")

    # 1. 解析 MOT 结果，得到每个 track 的所有检测
    track_detections = parse_mot_result(mot_result_path)

    # 2. 为每个轨迹生成候选图像并保存到 appearance_candidate 下对应子文件夹中
    save_candidate_images(track_detections, seq_folder, candidate_root)

    # 3. 创建最终 appearance 文件夹（存放每个轨迹筛选出的最佳候选图像）
    if not os.path.exists(final_appearance_folder):
        os.makedirs(final_appearance_folder)

    # 4. 设置设备、加载 Re-ID 模型及预处理
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=751,
        pretrained=True
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 5. 提取 target.jpg 的特征
    target_feat = extract_feature(target_img_path, model, transform, device)

    # 6. 对每个轨迹，从候选图像中选出与 target 相似度最高的那张，并保存到 final appearance 文件夹中
    track_best_similarity = {}  # 保存每个轨迹最佳候选与 target 的相似度
    for track_id in track_detections.keys():
        track_folder = os.path.join(candidate_root, str(track_id))
        if not os.path.exists(track_folder):
            print(f"轨迹 {track_id} 的候选文件夹不存在，跳过。")
            continue
        candidate_files = [os.path.join(track_folder, f) for f in os.listdir(track_folder)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(candidate_files) == 0:
            print(f"轨迹 {track_id} 没有候选图像，跳过。")
            continue
        best_sim = -1.0
        best_candidate = None
        for candidate_img in candidate_files:
            feat = extract_feature(candidate_img, model, transform, device)
            # 余弦相似度（归一化后直接点积）
            sim = torch.sum(feat * target_feat, dim=1).item()
            if sim > best_sim:
                best_sim = sim
                best_candidate = candidate_img
        if best_candidate is not None:
            final_save_path = os.path.join(final_appearance_folder, f"{track_id}.jpg")
            # 将最佳候选图像复制到 final appearance 文件夹
            img = cv2.imread(best_candidate)
            cv2.imwrite(final_save_path, img)
            print(f"轨迹 {track_id} 的最佳候选图像已保存到 {final_save_path}，相似度：{best_sim:.4f}")
            track_best_similarity[track_id] = best_sim
        else:
            print(f"轨迹 {track_id} 无有效候选图像。")

    # 7. 输出与 target.jpg 相似度最高的 5 个 track_id
    if len(track_best_similarity) == 0:
        print("没有轨迹候选图像处理成功。")
        return
    sorted_tracks = sorted(track_best_similarity.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_tracks[:5]
    print("\n与 target.jpg 最相似的 5 个轨迹 track_id：")
    for tid, sim in top5:
        print(f"track_id: {tid}, similarity: {sim:.4f}")
    return top5


def visualize_trajectory(track_id, start_time, video_fps):
    """
    可视化轨迹，并显示时间
    """
    # 1. 加载背景图
    background_img = cv2.imread("seq/000000.jpg")
    if background_img is None:
        print("无法加载背景图：seq/000000.jpg")
        return

    # 2. 解析 MOT 结果，得到轨迹数据
    mot_result_path = "output/UCMCTrack_mot_output.txt"  # 这个路径根据你的选择结果变化
    track_detections = parse_mot_result(mot_result_path)

    if track_id not in track_detections:
        print(f"没有找到轨迹 {track_id} 的数据")
        return

    # 3. 获取用户选择的轨迹的检测数据
    detections = track_detections[track_id]

    # 4. 创建用于绘制的图像（背景图）
    img = background_img.copy()

    # 5. 绘制轨迹
    prev_point = None
    for i, (frame_id, l, t, w, h) in enumerate(detections):
        # 计算时间
        timestamp = calculate_time(start_time, frame_id, video_fps)

        # 计算检测框中心点
        center_x = int(l + w / 2)
        center_y = int(t + h / 2)

        # 绘制连线
        if prev_point:
            cv2.line(img, prev_point, (center_x, center_y), (0, 0, 255), 2)

        # 每隔20个点打印时间
        if i % 20 == 0:
            cv2.putText(img, f"{timestamp}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        prev_point = (center_x, center_y)

    # 6. 显示结果
    cv2.imshow(f"轨迹 {track_id} 可视化", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

        # 信息显示框
        self.info_text = tk.Text(master, height=15, width=70)
        self.info_text.pack(pady=5)

    def upload_video(self):
        # 弹出文件选择对话框，选择视频文件（支持mp4、avi、mov等格式）
        selected_video = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov")]
        )
        if selected_video:
            # 创建存放视频文件的文件夹 "video"
            video_folder = "video"
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            # 将选择的视频复制到 video 目录下
            dest_path = os.path.join(video_folder, os.path.basename(selected_video))
            try:
                shutil.copy(selected_video, dest_path)
            except Exception as e:
                messagebox.showerror("错误", f"复制视频文件失败: {str(e)}")
                return
            self.video_path = dest_path
            self.info_text.insert(tk.END, f"视频已保存到：{self.video_path}\n")

    def upload_image(self):
        # 弹出文件选择对话框，选择图像文件（支持jpg、png、bmp等格式）
        selected_image = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if selected_image:
            # 创建存放图像文件的文件夹 "photo"
            photo_folder = "photo"
            if not os.path.exists(photo_folder):
                os.makedirs(photo_folder)

            # 打开图像文件
            try:
                img = Image.open(selected_image)
            except Exception as e:
                messagebox.showerror("错误", f"打开图像文件失败: {str(e)}")
                return

            # 判断是否是JPG格式，如果不是则转换为JPG
            if img.format != 'JPEG':
                # 转换为RGB（JPG不支持透明度）
                img = img.convert('RGB')

            # 将图像保存为 target.jpg
            dest_path = os.path.join(photo_folder, 'target.jpg')
            try:
                img.save(dest_path, 'JPEG')
            except Exception as e:
                messagebox.showerror("错误", f"保存图像文件失败: {str(e)}")
                return

            self.image_path = dest_path
            self.info_text.insert(tk.END, f"图片已保存到：{self.image_path}\n")

    def process_files(self):
        # 获取视频起始时间
        self.start_time = self.start_time_entry.get().strip()
        if not self.video_path or not self.image_path or not self.start_time:
            messagebox.showerror("错误", "请确保视频、图像文件以及视频起始时间都已提供。")
            return
        print("video_path", self.video_path)
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
            frame_filename = os.path.join(seq_folder, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        cap.release()

        self.info_text.insert(tk.END, f"视频已分解为 {frame_count} 帧，保存在文件夹 '{seq_folder}' 中。\n")
        self.info_text.insert(tk.END, f"视频起始时间：{self.start_time}\n")

        # ------------------------- 使用 YOLOv8 进行目标检测 -------------------------
        self.info_text.insert(tk.END, "正在加载检测模型，请稍候...\n")
        try:
            # ★★ 新的 YOLOv8 加载方式 ★★
            model = YOLO("yolov8s.pt")  # 若没下载过，会自动下载 yolov8s.pt
        except Exception as e:
            messagebox.showerror("错误", f"加载检测模型失败: {str(e)}")
            return
        self.info_text.insert(tk.END, "检测模型加载完成！\n")

        # 打开输出文件，写入检测结果
        output_file = "seq_det_res.txt"
        with open(output_file, "w") as f_out:
            # 获取 seq 文件夹下所有图像，按文件名排序
            frame_files = sorted(
                [file for file in os.listdir(seq_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            )
            for frame_file in frame_files:
                frame_path = os.path.join(seq_folder, frame_file)
                # 提取帧ID
                try:
                    frame_id_str = frame_file.split('.')[0]
                    frame_id = int(frame_id_str) + 1
                except Exception:
                    self.info_text.insert(tk.END, f"无法解析文件名 {frame_file} 的帧ID，跳过该文件。\n")
                    continue

                # ★★ 用 YOLOv8 对当前帧做检测，conf=0.5 表示置信度阈值 ★★
                results = model(frame_path, conf=0.5)

                # YOLOv8 返回一个列表，每张图对应一个 results[0]
                boxes = results[0].boxes
                # boxes.data 是 Nx6 的张量 [x1, y1, x2, y2, conf, cls]
                det_data = boxes.data.tolist()

                det_id = 1  # 检测目标编号从1开始
                for det in det_data:
                    x1, y1, x2, y2, conf, cls_id = det
                    width = x2 - x1
                    height = y2 - y1
                    # 构造输出行：{frame_id,det_id,l,t,w,h,conf,-1,-1,-1}
                    line = f"{frame_id},{det_id},{x1:.1f},{y1:.1f},{width:.1f},{height:.1f},{conf:.2f},-1,-1,-1\n"
                    f_out.write(line)
                    det_id += 1

                self.info_text.insert(tk.END, f"帧 {frame_id} 检测完成，共检测到 {det_id - 1} 个目标。\n")

        self.info_text.insert(tk.END, f"所有帧检测完成，结果已保存到 {output_file}\n")

        run_SORT()
        run_DeepSORT()
        run_UCMCTrack(self.img_width, self.img_height, self.video_fps)

        final_score_SORT, final_score_DeepSORT, final_score_UCMCTrack = cal_score()
        print(final_score_SORT, final_score_DeepSORT, final_score_UCMCTrack)
        best_res = ""
        if final_score_SORT >= final_score_DeepSORT and final_score_SORT >= final_score_UCMCTrack:
            best_res = "output/SORT_mot_output.txt"
        elif final_score_DeepSORT >= final_score_SORT and final_score_DeepSORT >= final_score_UCMCTrack:
            best_res = "output/DeepSORT_mot_output.txt"
        else:
            best_res = "output/UCMCTrack_mot_output.txt"

        top5 = re_id(best_res)

        # ------------------------- 弹出选择窗口 -------------------------
        # 新窗口用于显示 top5 的候选信息，让用户选择认为是真正 target 的轨迹
        selection_window = tk.Toplevel(self.master)
        selection_window.title("请选择正确的 target 轨迹")
        tk.Label(selection_window, text="请从下列轨迹中选择您认为是真正的 target:").pack(pady=5)

        # 用于存储用户的选择
        selected_track = tk.StringVar()
        selected_track.set("")  # 默认未选择

        # 在窗口中逐行显示每个候选轨迹的信息
        for idx, (track_id, sim) in enumerate(top5):
            img_path = os.path.join("appearance", f"{track_id}.jpg")
            if not os.path.exists(img_path):
                continue
            # 在每行显示文字信息及一个“查看图像”按钮
            frame = tk.Frame(selection_window, bd=2, relief=tk.RIDGE)
            frame.pack(padx=5, pady=5, fill="x")
            label_info = tk.Label(frame, text=f"Track {track_id}   Sim: {sim:.4f}")
            label_info.pack(side="left", padx=5)

            # 按钮用于调用 PIL.Image.show() 显示原始图像
            def show_image(path=img_path):
                img = Image.open(path)
                img.show()

            btn_show = tk.Button(frame, text="查看图像", command=show_image)
            btn_show.pack(side="left", padx=5)

            # 单选按钮供用户选择该候选轨迹
            rb = tk.Radiobutton(frame, text="选择", variable=selected_track, value=str(track_id))
            rb.pack(side="left", padx=5)

        def confirm_selection():
            chosen_track = selected_track.get()
            if not chosen_track:
                messagebox.showwarning("警告", "请先选择一个轨迹！")
                return
            messagebox.showinfo("选择结果", f"您选择的轨迹是：{chosen_track}")
            # 在 TODO 部分进行显示轨迹
            visualize_trajectory(int(chosen_track), self.start_time, self.video_fps)
            selection_window.destroy()

        confirm_btn = tk.Button(selection_window, text="确认选择", command=confirm_selection)
        confirm_btn.pack(pady=10)


if __name__ == "__main__":
    # 在处理文件之前，先清理文件夹
    clear_folders_and_files()

    set_random_seed(42)
    root = tk.Tk()  # 创建根窗口
    app = TRSApp(root)
    root.mainloop()

# 2025-03-09 08:00:00
