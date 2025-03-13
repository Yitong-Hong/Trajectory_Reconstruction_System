import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import cv2
import random

mot_res_path = "../output/mot17/val/MOT17-04-SDP.txt"
img_path = "../MOT17/train/MOT17-04-SDP/img1/000001.jpg"


def get_tracklet_for_instance(path):
    """
    从结果文件中读取所有轨迹信息，返回一个字典：
    {
        t_id: [[frame_id, center_x, center_y], ...],
        ...
    }
    """
    mp = {}
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip().split(',')
            frame_id = int(line[0])
            t_id = int(line[1])
            center_x = float(line[2]) + float(line[4]) / 2
            center_y = float(line[3]) + float(line[5]) / 2
            if t_id not in mp:
                mp[t_id] = []
            mp[t_id].append([frame_id, center_x, center_y])
    return mp


def visualize_single_track_on_single_image(track_data, ref_img_path, target_tid):
    """
    在单张静态图像 ref_img_path 上，只显示指定轨迹 (target_tid) 的线路图，
    并在每隔 10 帧时标记一次 frame_id。最终在轨迹末尾标记 t_id。
    """
    # 如果目标 t_id 不存在，给出提示
    if target_tid not in track_data:
        print(f"轨迹ID {target_tid} 不存在。")
        return

    # 读取单张参考图像
    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        print(f"无法读取图像: {ref_img_path}")
        return

    # 取出 target_tid 对应的所有点，并按 frame_id 升序排序
    points = track_data[target_tid]
    points_sorted = sorted(points, key=lambda x: x[0])

    # 给每条轨迹一个随机颜色，也可写成固定色，如 (0, 255, 0)
    color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

    # 逐点连线，并在 frame_id 能被 10 整除时标记一次
    for i in range(len(points_sorted)):
        frame_id_i, x_i, y_i = points_sorted[i]

        # 与上一个点连线
        if i > 0:
            frame_id_prev, x_prev, y_prev = points_sorted[i - 1]
            cv2.line(
                ref_img,
                (int(x_prev), int(y_prev)),  # 上一个点
                (int(x_i), int(y_i)),  # 当前点
                color,
                2
            )
        # 每隔10帧时在该点标出 frame_id
        if frame_id_i % 10 == 0:
            cv2.putText(
                ref_img,
                str(frame_id_i),
                (int(x_i), int(y_i)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # 轨迹终点标上 t_id
    _, x_end, y_end = points_sorted[-1]
    cv2.putText(
        ref_img,
        f"t_id: {target_tid}",
        (int(x_end), int(y_end)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # 显示图片
    cv2.imshow(f"Track of t_id={target_tid}", ref_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_gui_app():
    """
    使用 Tkinter 创建一个简单的 GUI：
    - 显示所有可用的 t_id 范围
    - 提供一个下拉菜单让用户选择某个 t_id
    - 点击 "可视化" 按钮后，调用可视化函数
    """
    # 先读取轨迹数据并获取所有 t_id
    track_data = get_tracklet_for_instance(mot_res_path)
    all_ids = sorted(track_data.keys())
    if not all_ids:
        messagebox.showerror("Error", "结果文件中没有任何轨迹！")
        return

    # 创建主窗口
    root = tk.Tk()
    root.title("MOT Track ID Visualizer")

    # 在界面上显示总的 t_id 信息
    info_label = tk.Label(
        root,
        text=f"在 {mot_res_path} 中，共找到 {len(all_ids)} 个轨迹ID。\n"
             f"范围：[ {all_ids[0]} ~ {all_ids[-1]} ]"
    )
    info_label.pack(pady=10)

    # 提示文字
    select_label = tk.Label(root, text="请选择一个轨迹ID进行可视化：")
    select_label.pack()

    # 使用下拉菜单(ComboBox)以显示所有可用 t_id
    t_id_var = tk.StringVar(root)
    t_id_var.set(str(all_ids[0]))  # 默认先选第一个
    combo = ttk.Combobox(root, textvariable=t_id_var, values=[str(x) for x in all_ids], state='readonly')
    combo.pack()

    # 点击按钮后，调用可视化函数
    def on_visualize():
        selected_tid_str = t_id_var.get()
        if not selected_tid_str.isdigit():
            messagebox.showwarning("Warning", "请选择或输入一个整数轨迹ID!")
            return

        selected_tid = int(selected_tid_str)
        if selected_tid not in all_ids:
            messagebox.showwarning("Warning", f"轨迹ID {selected_tid} 不在可用范围内!")
            return

        # 调用可视化函数
        visualize_single_track_on_single_image(track_data, img_path, selected_tid)

    # 按钮
    visualize_button = tk.Button(root, text="可视化", command=on_visualize)
    visualize_button.pack(pady=10)

    # 进入事件循环
    root.mainloop()


if __name__ == "__main__":
    run_gui_app()
