import os
import sys
import time
import cv2
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt6.QtCore import QPoint, QObject
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtWidgets import (QApplication, QWidget, QFileDialog, QMessageBox, QComboBox,
                             QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, 
                             QCheckBox, QProgressBar, QGroupBox, QGridLayout, QSizePolicy, QScrollArea)
from ultralytics import YOLO
from qt_material import apply_stylesheet
from mainWindow import Ui_Form

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def get_fly_color(unique_id):
    """
    基于果蝇唯一ID生成稳定的BGR颜色（OpenCV使用BGR格式）
    :param unique_id: 果蝇的track_id（视频）或序号（图片）
    :return: (B, G, R) 颜色元组
    """
    # 固定哈希因子，确保同一ID生成相同颜色，不同ID颜色差异明显
    b = (unique_id * 137) % 256  # 蓝色通道
    g = (unique_id * 27) % 256   # 绿色通道
    r = (unique_id * 197) % 256  # 红色通道
    return (int(b), int(g), int(r))
def calculate_circle_from_three_points(p1, p2, p3):
    """通过三个点计算圆的圆心和半径"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # 检查三点是否共线
    if (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1):
        return None  # 三点共线，无法确定圆
    
    # 计算圆心坐标
    A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    B = (x1**2 + y1**2) * (y3 - y2) + (x2**2 + y2**2) * (y1 - y3) + (x3**2 + y3**2) * (y2 - y1)
    C = (x1**2 + y1**2) * (x2 - x3) + (x2**2 + y2**2) * (x3 - x1) + (x3**2 + y3**2) * (x1 - x2)
    
    if A == 0:
        return None
    
    center_x = -B / (2 * A)
    center_y = -C / (2 * A)
    
    # 计算半径
    radius = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    
    return (int(center_x), int(center_y), int(radius))


class DetectionWorker(QObject):
    """在后台线程中执行视频/摄像头检测任务，支持轨迹绘制和ROI区域检测"""
    frame_processed = pyqtSignal(QImage, np.ndarray)
    finished = pyqtSignal()
    detection_info = pyqtSignal(str)
    video_saved = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)  # (current_frame, total_frames)

    def __init__(self, video_source, model_path, conf_value, font_size, roi_circle=None, 
                 output_path=None, show_trace=True, video_speed=1.0, contrast_value=0):
        super().__init__()
        self.video_source = video_source
        self.model_path = model_path
        self.conf_value = conf_value
        self.font_size = font_size
        self.is_running = True
        self.model = None
        self.capture = None
        self.trajectories = {}
        self.max_trace_length = 30
        self.roi_circle = roi_circle
        self.output_path = output_path
        self.out = None
        self.coordinates_data = []
        self.video_speed = video_speed
        self.show_trace = show_trace
        self.contrast_value = contrast_value
        self.total_frames = 0

    def run(self):


        """线程执行的入口点"""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.finished.emit()
            return

        if isinstance(self.video_source, str) and os.path.exists(self.video_source):
            self.capture = cv2.VideoCapture(self.video_source)
        elif isinstance(self.video_source, int):
            self.capture = cv2.VideoCapture(self.video_source)
        else:
            print(f"无效的视频源: {self.video_source}")
            self.finished.emit()
            return

        if not self.capture.isOpened():
            print(f"无法打开视频源: {self.video_source}")
            self.finished.emit()
            return

        # 获取视频总帧数和FPS
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total_frames if total_frames > 0 else 0
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 默认帧率
        
        # 计算每帧的基础延迟时间（秒）
        base_frame_delay = 1.0 / fps
        print(f"视频FPS: {fps}, 基础帧延迟: {base_frame_delay:.4f}秒")

        # 初始化视频写入器
        if self.output_path and isinstance(self.video_source, str):
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count = 0
        id_to_sequence = {}
        current_sequence = 1

        # 存储每个track_id的轨迹点（新增：同时记录颜色）
        self.track_history = {}  # 格式: {track_id: [(x,y), (x,y), ...]}

        while self.is_running:
            ret, frame = self.capture.read()
            if not ret:
                break

            frame_count += 1
            frame_data = {"frame": frame_count, "flies": []}

            # 发送进度更新
            if self.total_frames > 0:
                self.progress_update.emit(frame_count, self.total_frames)

            # 应用对比度调节
            if self.contrast_value != 0:
                alpha = (100 + self.contrast_value) / 100.0
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

            # 应用ROI区域裁剪
            if self.roi_circle:
                x, y, radius = self.roi_circle
                h, w = frame.shape[:2]
                x = max(0, min(int(x), w - 1))
                y = max(0, min(int(y), h - 1))
                radius = min(int(radius), min(x, w - x, y, h - y))

                mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)
                frame = cv2.bitwise_and(frame, frame, mask=mask)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.track(frame_rgb,conf=self.conf_value,persist=True,tracker="bytetrack.yaml"
)[0]

            trace_layer = np.zeros_like(frame_rgb, dtype=np.uint8)

            coordinates_info = f"第 {frame_count} 帧检测到 {len(results.boxes)} 只果蝇:"
            print("\n" + "=" * 30)
            print(coordinates_info)
            self.detection_info.emit(coordinates_info)

            if results.boxes is not None and len(results.boxes) > 0 and results.boxes.id is not None:
                for box, track_id in zip(results.boxes, results.boxes.id):
                    track_id = int(track_id.item())

                    if track_id not in id_to_sequence:
                        id_to_sequence[track_id] = current_sequence
                        current_sequence += 1

                    sequence_number = id_to_sequence[track_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    info = f"  果蝇 {sequence_number}"
                    print(info)
                    self.detection_info.emit(info)

                    frame_data["flies"].append({
                        "id": sequence_number,
                        "center_x": round(center_x, 2),
                        "center_y": round(center_y, 2)
                    })

                    # 绘制轨迹
                    if self.show_trace:
                        if track_id not in self.trajectories:
                            self.trajectories[track_id] = []

                        self.trajectories[track_id].append((int(center_x), int(center_y)))

                        if len(self.trajectories[track_id]) > self.max_trace_length:
                            self.trajectories[track_id].pop(0)

                        if len(self.trajectories[track_id]) > 1:
                            points = np.array(self.trajectories[track_id], np.int32)
                            color_seed = track_id * 100
                            color = (
                                (color_seed * 137) % 256,
                                (color_seed * 27) % 256,
                                (color_seed * 197) % 256
                            )
                            cv2.polylines(trace_layer, [points], isClosed=False, color=color, thickness=2)

            self.coordinates_data.append(frame_data)
            print("=" * 30)

            # 绘制检测结果
            annotated_frame = frame_rgb.copy()
            if results.boxes is not None and len(results.boxes) > 0 and results.boxes.id is not None:
                for box, track_id in zip(results.boxes, results.boxes.id):
                    track_id = int(track_id.item())
                    sequence_number = id_to_sequence.get(track_id, 0)

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    conf = box.conf[0].item()

                    # 复用轨迹的颜色（与轨迹线同色）
                    color_seed = track_id * 100
                    color = (
                        (color_seed * 137) % 256,
                        (color_seed * 27) % 256,
                        (color_seed * 197) % 256
                    )

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.font_size)
                    label = f"fly {sequence_number} ({center_x:.1f},{center_y:.1f}) {conf:.2f}"

                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - text_height - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

            # 绘制ROI区域
            if self.roi_circle:
                x, y, radius = self.roi_circle
                cv2.circle(annotated_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

            # 合并轨迹线
            final_frame = cv2.addWeighted(annotated_frame, 1.0, trace_layer, 0.7, 0)

            # 保存视频
            final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            if self.out:
                self.out.write(final_frame_bgr)

            # 发送帧
            h, w, ch = final_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(final_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_processed.emit(qt_image, final_frame_bgr)

            # 倍速控制：根据视频FPS和倍速计算延迟
            # 延迟 = 基础帧延迟 / 倍速
            actual_delay = base_frame_delay / self.video_speed
            time.sleep(actual_delay)

        # 保存坐标数据
        if self.coordinates_data and self.output_path:
            txt_path = os.path.splitext(self.output_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for frame_data in self.coordinates_data:
                    f.write(f"第 {frame_data['frame']} 帧:\n")
                    for fly in frame_data['flies']:
                        f.write(f"  果蝇 {fly['id']}: 中心点({fly['center_x']}, {fly['center_y']})\n")
                    f.write("\n")

        self.cleanup()
        self.finished.emit()
        if self.output_path:
            self.video_saved.emit(self.output_path)

    def stop(self):
        """停止线程"""
        self.is_running = False

    def cleanup(self):
        """释放资源"""
        if self.capture:
            self.capture.release()
        if self.out:
            self.out.release()


class YOLOv10App(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        # 初始化变量
        self.video_speed = 1.0
        self.show_trace = True
        self.model = None
        self.confValue = 0.5
        self.fontSize = 2
        self.image_path = None
        self.video_path = None
        self.processed_image = None
        self.original_image = None  # 保存原始图像用于实时对比度调整
        self.contrast_value = 0
        self.roi_circle = None
        self.roi_points = []
        self.is_drawing_roi = False
        self.worker = None
        self.thread = None
        
        # 1. 清除旧布局
        old_layout = self.layout()
        if old_layout:
            QWidget().setLayout(old_layout) 
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 2. 创建侧边栏滚动区域
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setFixedWidth(295)  # 考虑滚动条宽度
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setContentsMargins(5, 5, 5, 5)
        self.sidebar_layout.setSpacing(8)
        
        self.sidebar_scroll.setWidget(self.sidebar_widget)

        # 3. 创建显示区
        self.display_container = QWidget()
        self.display_layout = QVBoxLayout(self.display_container)
        self.display_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self.sidebar_scroll)
        main_layout.addWidget(self.display_container, 1)

        self.setup_sidebar_ui()

        self.display_label = QLabel("请选择图片或视频进行检测")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet("background-color: #e6e6e6; border: 2px dashed #ccc; border-radius: 10px;")
        self.display_label.setMinimumSize(400, 300)
        self.display_label.setMouseTracking(True)
        self.display_label.mousePressEvent = self.on_mouse_press
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")  # 只显示百分比
        self.progress_bar.setFixedHeight(25)
        
        self.display_layout.addWidget(self.display_label, 1)
        self.display_layout.addWidget(self.progress_bar)

        # 加载模型
        try:
            self.load_models(r'C:\Users\85212\runs\detect\train28\weights')
        except:
            pass

        # 绑定信号
        self.bind_signals()
        
        # 初始化数值
        self.ui.horizontalSlider_conf.setValue(int(self.confValue * 100))
        self.ui.dial_fontsize.setValue(self.fontSize)
        self.updateConfValue(self.ui.horizontalSlider_conf.value())
        self.updateFontsize(self.ui.dial_fontsize.value())
        
        # 设置窗口大小
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

    def setup_sidebar_ui(self):
        """内核修复：手动组装侧边栏，确保所有控件可见"""
        
        # --- 1. 模型选择组 (新建) ---
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        # 把下拉框从原来的位置“偷”出来
        self.ui.comboBox_choseModel.setParent(None) 
        self.ui.comboBox_choseModel.setMinimumHeight(30)
        model_layout.addWidget(self.ui.comboBox_choseModel)
        
        self.sidebar_layout.addWidget(model_group)
        
        # --- 2. 参数设置组 (置信度 + 字体) (新建) ---
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)
        
        # 置信度
        conf_layout = QHBoxLayout()
        label_conf_title = QLabel("置信度:")
        self.ui.label_conf.setParent(None) # 具体的数值标签
        
        self.ui.horizontalSlider_conf.setParent(None)
        self.ui.horizontalSlider_conf.setOrientation(Qt.Orientation.Horizontal)
        self.ui.horizontalSlider_conf.setFixedHeight(20)
        
        conf_layout.addWidget(label_conf_title)
        conf_layout.addWidget(self.ui.horizontalSlider_conf)
        conf_layout.addWidget(self.ui.label_conf)
        param_layout.addLayout(conf_layout)
        
        # 字体大小 (原本是 Dial 旋钮，这里为了美观可以和文本放在一行)
        font_layout = QHBoxLayout()
        label_font_title = QLabel("结果字号:")
        self.ui.dial_fontsize.setParent(None)
        self.ui.dial_fontsize.setFixedSize(40, 40) # 限制旋钮大小
        self.ui.label_fontsize.setParent(None)
        
        font_layout.addWidget(label_font_title)
        font_layout.addWidget(self.ui.dial_fontsize)
        font_layout.addWidget(self.ui.label_fontsize)
        font_layout.addStretch() # 靠左对齐
        param_layout.addLayout(font_layout)
        
        self.sidebar_layout.addWidget(param_group)

        # --- 3. 功能按钮组 (新建) ---
        function_group = QGroupBox("功能操作")
        function_layout = QVBoxLayout(function_group)
        function_layout.setSpacing(5)
        
        btns = [
            (self.ui.pushButton_selectImage, "1. 选择图片"),
            (self.ui.pushButton_detectImage, "2. 检测图片"),
            (self.ui.pushButton_4, "3. 选择视频"),
            (self.ui.pushButton_3, "4. 开始视频检测"),
            (self.ui.pushButton_5, "5. 停止检测"),
            (self.ui.pushButton_9, "6. 打开摄像头"),
            (self.ui.pushButton_10, "7. 停止摄像头"),
            (self.ui.pushButton_setRoi, "8. 设置检测区域(ROI)")
        ]
        
        for btn, text in btns:
            btn.setText(text)
            btn.setParent(None)
            btn.setMinimumHeight(30)
            btn.setMaximumHeight(40)
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed
            )
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 11px;
                    padding: 3px;
                }
            """)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            function_layout.addWidget(btn)
            
        self.sidebar_layout.addWidget(function_group)

        # --- 4. 对比度设置 (搬运) ---
        contrast_group = QGroupBox("图像增强")
        contrast_layout = QVBoxLayout(contrast_group)
        
        c_layout = QHBoxLayout()
        c_label = QLabel("对比度:")
        self.ui.horizontalSlider1.setParent(None)
        self.ui.horizontalSlider1.setOrientation(Qt.Orientation.Horizontal)
        self.ui.horizontalSlider1.setMinimum(-100)
        self.ui.horizontalSlider1.setMaximum(100)
        self.ui.horizontalSlider1.setValue(0)
        self.ui.label_5.setParent(None)
        self.ui.label_5.setText("0")
        
        c_layout.addWidget(c_label)
        c_layout.addWidget(self.ui.horizontalSlider1)
        c_layout.addWidget(self.ui.label_5)
        contrast_layout.addLayout(c_layout)
        
        self.sidebar_layout.addWidget(contrast_group)

        # --- 5. 视频控制 (自定义的新控件) ---
        video_group = QGroupBox("视频控制")
        v_layout = QVBoxLayout(video_group)
        
        # 轨迹开关
        self.trace_checkbox = QCheckBox("显示果蝇轨迹")
        self.trace_checkbox.setChecked(True)
        self.trace_checkbox.setMinimumHeight(25)
        self.trace_checkbox.stateChanged.connect(self.on_trace_toggled)
        v_layout.addWidget(self.trace_checkbox)
        
        # 倍速控制
        h_layout = QHBoxLayout()
        speed_label = QLabel("播放倍速:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "1.5x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        
        h_layout.addWidget(speed_label)
        h_layout.addWidget(self.speed_combo)
        v_layout.addLayout(h_layout)
        
        # ROI 提示
        self.roi_info_label = QLabel("提示: 点击[设置检测区域]后\n在右图点击3次确定圆形")
        self.roi_info_label.setStyleSheet("color: #666; font-size: 11px;")
        self.roi_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_layout.addWidget(self.roi_info_label)

        self.sidebar_layout.addWidget(video_group)
        
        # 底部弹簧
        self.sidebar_layout.addStretch()

    def setup_custom_ui(self):
        pass
    
    def bind_signals(self):
        """绑定所有UI信号"""
        self.ui.horizontalSlider_conf.valueChanged.connect(self.updateConfValue)
        self.ui.dial_fontsize.valueChanged.connect(self.updateFontsize)
        self.ui.horizontalSlider1.valueChanged.connect(self.update_contrast)
        
        self.ui.pushButton_selectImage.clicked.connect(self.selectImage)
        self.ui.pushButton_detectImage.clicked.connect(self.detectImage)
        self.ui.pushButton_4.clicked.connect(self.selectVideo)
        self.ui.pushButton_3.clicked.connect(self.startVideoDetection)
        self.ui.pushButton_5.clicked.connect(self.stopDetection)
        self.ui.pushButton_9.clicked.connect(self.startCameraDetection)
        self.ui.pushButton_10.clicked.connect(self.stopDetection)
        self.ui.pushButton_setRoi.clicked.connect(self.toggle_roi_drawing)
    
    def on_trace_toggled(self, state):
        """轨迹显示切换"""
        self.show_trace = (state == Qt.CheckState.Checked.value)
        print(f"轨迹显示: {'开启' if self.show_trace else '关闭'}")
    
    def on_speed_changed(self, speed_text):
        """倍速变化回调"""
        self.video_speed = float(speed_text.replace("x", ""))
        print(f"视频速度: {self.video_speed}x")
        
        # 如果视频正在播放，实时更新Worker的速度
        if self.worker is not None:
            self.worker.video_speed = self.video_speed
    
    def update_contrast(self, value):
        """更新对比度并实时预览"""
        self.contrast_value = value
        self.ui.label_5.setText(str(value))
        
        # 如果已加载图像，实时应用对比度
        if self.original_image is not None:
            self.apply_contrast_to_display()
        
        # 如果视频正在播放，实时更新Worker的对比度
        if self.worker is not None:
            self.worker.contrast_value = self.contrast_value
    
    def apply_contrast_to_display(self):
        """实时应用对比度到显示图像"""
        if self.original_image is None:
            return
        
        # 复制原始图像
        img = self.original_image.copy()
        
        # 应用对比度
        if self.contrast_value != 0:
            alpha = (100 + self.contrast_value) / 100.0
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        # 如果有ROI，绘制ROI圆圈（不裁剪，只显示边界）
        if self.roi_circle:
            x, y, radius = self.roi_circle
            cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
            # 绘制圆心
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        
        # 转换为RGB并显示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        self.display_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )
    
    def toggle_roi_drawing(self):
        """切换ROI绘制模式（三点确定圆）"""
        self.is_drawing_roi = not self.is_drawing_roi
        if self.is_drawing_roi:
            self.ui.pushButton_setRoi.setText("结束绘制ROI")
            self.roi_circle = None
            self.roi_points = []
            QMessageBox.information(self, "绘制ROI", "请在显示区域点击3个点来确定圆形ROI")
        else:
            self.ui.pushButton_setRoi.setText("框选范围")
            if self.roi_circle:
                x, y, r = self.roi_circle
                QMessageBox.information(self, "ROI设置完成", 
                                      f"已设置圆形ROI：中心({x},{y})，半径{r}")
            self.roi_points = []
    
    def on_mouse_press(self, event):
        """鼠标点击事件（三点确定圆）"""
        if not self.is_drawing_roi or event.button() != Qt.MouseButton.LeftButton:
            return
        
        # 获取点击位置
        widget_x, widget_y = event.pos().x(), event.pos().y()
        img_x, img_y = self.map_widget_to_image_coords(widget_x, widget_y)
        
        # 添加点到列表
        self.roi_points.append((img_x, img_y))
        print(f"已选择第 {len(self.roi_points)} 个点: ({img_x}, {img_y})")
        
        # 在图像上标记点
        self._draw_roi_points()
        
        # 如果已选择3个点，计算圆
        if len(self.roi_points) == 3:
            circle_data = calculate_circle_from_three_points(
                self.roi_points[0], 
                self.roi_points[1], 
                self.roi_points[2]
            )
            
            if circle_data:
                self.roi_circle = circle_data
                x, y, r = circle_data
                print(f"ROI圆形已确定：中心({x}, {y})，半径{r}")
                QMessageBox.information(self, "ROI确定", 
                                      f"圆形ROI已设定\n中心: ({x}, {y})\n半径: {r}")
                self._draw_roi_circle()
                # 自动结束绘制模式
                self.is_drawing_roi = False
                self.ui.pushButton_setRoi.setText("框选范围")
            else:
                QMessageBox.warning(self, "ROI无效", "三个点共线，无法确定圆形，请重新选择")
                self.roi_points = []
    
    def _draw_roi_points(self):
        """在图像上绘制已选择的ROI点"""
        if self.original_image is None:
            return
        
        # 使用应用对比度后的图像
        img = self.original_image.copy()
        if self.contrast_value != 0:
            alpha = (100 + self.contrast_value) / 100.0
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        # 转换为QPixmap
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        painter = QPainter(pixmap)
        
        # 绘制已选择的点
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.red, 8, Qt.PenStyle.SolidLine))
        for point in self.roi_points:
            painter.drawPoint(QPoint(point[0], point[1]))
        
        # 如果有两个点，绘制连线
        if len(self.roi_points) >= 2:
            painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.blue, 2, Qt.PenStyle.DashLine))
            for i in range(len(self.roi_points) - 1):
                p1 = QPoint(self.roi_points[i][0], self.roi_points[i][1])
                p2 = QPoint(self.roi_points[i+1][0], self.roi_points[i+1][1])
                painter.drawLine(p1, p2)
        
        painter.end()
        
        self.display_label.setPixmap(pixmap.scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
    
    def _draw_roi_circle(self):
        """在图像上绘制最终确定的ROI圆"""
        if self.original_image is None or not self.roi_circle:
            return
        
        # 使用应用对比度后的图像
        img = self.original_image.copy()
        if self.contrast_value != 0:
            alpha = (100 + self.contrast_value) / 100.0
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        # 转换为QPixmap
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        painter = QPainter(pixmap)
        
        x, y, radius = self.roi_circle
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.green, 3, Qt.PenStyle.SolidLine))
        painter.drawEllipse(QPoint(x, y), radius, radius)
        
        # 绘制圆心
        painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.red, 8, Qt.PenStyle.SolidLine))
        painter.drawPoint(QPoint(x, y))
        
        painter.end()
        
        self.display_label.setPixmap(pixmap.scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
    
    def map_widget_to_image_coords(self, widget_x, widget_y):
        """将控件坐标映射到图像坐标"""
        if not self.image_path:
            return widget_x, widget_y
        
        original_pixmap = QPixmap(self.image_path)
        img_w = original_pixmap.width()
        img_h = original_pixmap.height()
        
        label_w = self.display_label.width()
        label_h = self.display_label.height()
        
        scale = min(label_w / img_w, label_h / img_h)
        scaled_w = img_w * scale
        scaled_h = img_h * scale
        
        offset_x = (label_w - scaled_w) / 2
        offset_y = (label_h - scaled_h) / 2
        
        image_x = (widget_x - offset_x) / scale
        image_y = (widget_y - offset_y) / scale
        
        image_x = max(0, min(image_x, img_w - 1))
        image_y = max(0, min(image_y, img_h - 1))
        
        return int(image_x), int(image_y)
    
    def load_models(self, directory):
        """加载模型列表"""
        try:
            if not os.path.exists(directory):
                QMessageBox.warning(self, '目录错误', '未找到模型目录')
                return
            
            model_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
            self.ui.comboBox_choseModel.addItems(model_files)
            if model_files:
                self.ui.comboBox_choseModel.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载模型失败: {e}')
    
    def updateConfValue(self, value):
        """更新置信度"""
        self.confValue = value / 100.0
        self.ui.label_conf.setText(f"{self.confValue:.2f}")
    
    def updateFontsize(self, value):
        """更新字体大小"""
        self.fontSize = value
        self.ui.label_fontsize.setText(f"{self.fontSize:d}")
    
    def getCurrentModelPath(self):
        """获取当前选择的模型路径"""
        currModelName = self.ui.comboBox_choseModel.currentText()
        if not currModelName:
            QMessageBox.warning(self, '模型错误', '请选择检测模型')
            return None
        return os.path.join(r'C:\Users\85212\runs\detect\train28\weights', currModelName)
    
    def selectImage(self):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.image_path = file_path
            self.roi_circle = None
            self.roi_points = []
            self.is_drawing_roi = False
            self.ui.pushButton_setRoi.setText("框选范围")
            
            # 读取并保存原始图像
            self.original_image = cv2.imread(file_path)
            
            # 应用当前对比度显示
            self.apply_contrast_to_display()
    
    def detectImage(self):
        """检测图片"""
        if self.image_path is None:
            QMessageBox.warning(self, '数据错误', '未导入待检测图片')
            return
        
        model_path = self.getCurrentModelPath()
        if model_path is None:
            return
        
        self.ui.pushButton_detectImage.setEnabled(False)
        
        try:
            model = YOLO(model_path)
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("无法加载图像")
            
            # 应用对比度
            if self.contrast_value != 0:
                alpha = (100 + self.contrast_value) / 100.0
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            
            # 应用ROI
            roi_applied = False
            if self.roi_circle:
                x, y, radius = self.roi_circle
                h, w = img.shape[:2]
                x = max(0, min(int(x), w - 1))
                y = max(0, min(int(y), h - 1))
                radius = min(int(radius), min(x, w - x, y, h - y))
                
                mask = np.zeros(img.shape[:2], np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)
                img = cv2.bitwise_and(img, img, mask=mask)
                roi_applied = True
            
            results = model(img, conf=self.confValue)[0]
            
            # 收集坐标
            coordinates_data = []
            print("=" * 30)
            print(f"图片检测到 {len(results.boxes)} 只果蝇:")
            
            if results.boxes is not None and len(results.boxes) > 0:
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    info = f"  果蝇 {i + 1}"
                    print(info)
                    coordinates_data.append({
                        "id": i + 1,
                        "center_x": round(center_x, 2),
                        "center_y": round(center_y, 2)
                    })
            
            print("=" * 30)
            
            # 绘制结果
            annotated_img = img.copy()
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                conf = box.conf[0].item()

                # 基于果蝇ID生成唯一颜色（BGR格式）
                color_seed = i + 1  # 使用果蝇序号作为种子
                color = (
                    (color_seed * 137) % 256,  # B通道
                    (color_seed * 27) % 256,  # G通道
                    (color_seed * 197) % 256  # R通道
                )
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, self.fontSize)
                label = f"fly {i + 1} ({center_x:.1f},{center_y:.1f}) {conf:.2f}"
                
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    annotated_img,
                    (x1, y1 - text_height - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
            
            # 绘制ROI
            if roi_applied and self.roi_circle:
                x, y, radius = self.roi_circle
                cv2.circle(annotated_img, (x, y), radius, (0, 255, 0), 2)
            
            self.processed_image = annotated_img
            self.save_image_results(annotated_img, coordinates_data)
            
            # 显示结果
            qImage = QImage(annotated_img.data, annotated_img.shape[1], annotated_img.shape[0],
                          QImage.Format.Format_BGR888)
            self.display_label.setPixmap(
                QPixmap.fromImage(qImage).scaled(
                    self.display_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
        except Exception as e:
            QMessageBox.critical(self, '检测错误', f'发生错误: {e}')
        finally:
            self.ui.pushButton_detectImage.setEnabled(True)
    
    def save_image_results(self, image, coordinates):
        """保存图片检测结果"""
        default_name = os.path.splitext(os.path.basename(self.image_path))[0]
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果",
            f"{default_name}_detected.jpg",
            "JPEG Images (*.jpg);;PNG Images (*.png)"
        )
        
        if save_path:
            cv2.imwrite(save_path, image)
            
            txt_path = os.path.splitext(save_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"图片检测到 {len(coordinates)} 只果蝇:\n")
                for fly in coordinates:
                    f.write(f"  果蝇 {fly['id']}: 中心点({fly['center_x']}, {fly['center_y']})\n")
            
            QMessageBox.information(self, "保存成功",
                                  f"检测结果已保存至:\n{save_path}\n坐标数据已保存至:\n{txt_path}")
    
    # ========== 视频检测 ==========
    def selectVideo(self):
        """选择视频"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if video_path:
            self.video_path = video_path
            self.roi_circle = None
            self.roi_points = []
            self.is_drawing_roi = False
            self.ui.pushButton_setRoi.setText("框选范围")
            
            try:
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    # 保存原始首帧
                    self.original_image = frame.copy()
                    
                    import tempfile
                    temp_path = os.path.join(tempfile.gettempdir(), "video_first_frame.jpg")
                    cv2.imwrite(temp_path, frame)
                    self.image_path = temp_path
                    
                    # 应用当前对比度显示
                    self.apply_contrast_to_display()
                else:
                    self.display_label.setText(f"已选择视频: {os.path.basename(video_path)}")
                    self.original_image = None
                cap.release()
            except Exception as e:
                print(f"读取视频失败: {e}")
                self.display_label.setText(f"已选择视频: {os.path.basename(video_path)}")
    
    def startVideoDetection(self):
        """开始视频检测"""
        if self.video_path is None:
            QMessageBox.warning(self, '数据错误', '请先选择视频文件')
            return
        
        default_name = os.path.splitext(os.path.basename(self.video_path))[0]
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存视频检测结果",
            f"{default_name}_detected.mp4",
            "MP4 Videos (*.mp4);;AVI Videos (*.avi)"
        )
        
        if save_path:
            self._start_detection(self.video_path, save_path)
    
    def startCameraDetection(self):
        """开始摄像头检测"""
        self._start_detection(0)
    
    def _start_detection(self, source, output_path=None):
        """启动检测线程"""
        model_path = self.getCurrentModelPath()
        if model_path is None:
            return
        
        self.stopDetection()
        
        # 清除原始图像，防止干扰视频播放
        self.original_image = None
        
        self.worker = DetectionWorker(
            video_source=source,
            model_path=model_path,
            conf_value=self.confValue,
            font_size=self.fontSize,
            roi_circle=self.roi_circle,
            output_path=output_path,
            show_trace=self.show_trace,
            video_speed=self.video_speed,
            contrast_value=self.contrast_value
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.frame_processed.connect(self.updateVideoFrame)
        self.worker.progress_update.connect(self.updateProgress)
        self.worker.finished.connect(self.onDetectionFinished)
        self.worker.video_saved.connect(self.onVideoSaved)
        
        self.thread.start()
        
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton_9.setEnabled(False)
        self.ui.pushButton_5.setEnabled(True)
        self.ui.pushButton_10.setEnabled(True)
        self.progress_bar.setValue(0)
        self.display_label.setText("检测中...")
    
    @pyqtSlot(QImage, np.ndarray)
    def updateVideoFrame(self, qImage, frame):
        """更新视频帧显示"""
        if qImage.isNull():
            return
        
        self.processed_image = frame
        pixmap = QPixmap.fromImage(qImage)
        self.display_label.setPixmap(pixmap.scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        self.display_label.update()
    
    @pyqtSlot(int, int)
    def updateProgress(self, current, total):
        """更新进度条"""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
    
    def onVideoSaved(self, path):
        """视频保存完成"""
        txt_path = os.path.splitext(path)[0] + ".txt"
        QMessageBox.information(self, "保存成功",
                              f"视频检测结果已保存至:\n{path}\n坐标数据已保存至:\n{txt_path}")
    
    def stopDetection(self):
        """停止检测"""
        if self.worker:
            self.worker.stop()
    
    @pyqtSlot()
    def onDetectionFinished(self):
        """检测完成"""
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.worker = None
        self.thread = None
        
        self.ui.pushButton_3.setEnabled(True)
        self.ui.pushButton_9.setEnabled(True)
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_10.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        # 摄像头模式保存最后一帧
        if isinstance(getattr(self, 'video_source', None), int) and self.processed_image is not None:
            reply = QMessageBox.question(
                self, "保存摄像头帧",
                "是否保存最后一帧图像?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "保存摄像头帧",
                    "camera_frame.jpg",
                    "JPEG Images (*.jpg);;PNG Images (*.png)"
                )
                if save_path:
                    cv2.imwrite(save_path, self.processed_image)
                    QMessageBox.information(self, "保存成功", f"图像已保存至:\n{save_path}")


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="light_blue.xml")
    ex = YOLOv10App()
    ex.show()
    sys.exit(app.exec())
