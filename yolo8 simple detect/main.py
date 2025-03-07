import cv2
import time
import numpy as np

from ultralytics import YOLO

from patched_yolo_infer import (
    visualize_results_usual_yolo_inference,
)

class FPS_Counter:
    def __init__(self, calc_time_perion_N_frames: int) -> None:
        """Счетчик FPS по ограниченным участкам видео (скользящему окну).

        Args:
            calc_time_perion_N_frames (int): количество фреймов окна подсчета статистики.
        """
        self.time_buffer = []
        self.calc_time_perion_N_frames = calc_time_perion_N_frames

    def calc_FPS(self) -> float:
        """Производит рассчет FPS по нескольким кадрам видео.

        Returns:
            float: значение FPS.
        """
        time_buffer_is_full = len(self.time_buffer) == self.calc_time_perion_N_frames
        t = time.time()
        self.time_buffer.append(t)

        if time_buffer_is_full:
            self.time_buffer.pop(0)
            fps = len(self.time_buffer) / (self.time_buffer[-1] - self.time_buffer[0])
            return np.round(fps, 2)
        else:
            return 0.0


def calc_and_show_fps(frame, fps_counter):
    """
    Вычисляет и отображает FPS на кадре.

    Args:
        frame (numpy.ndarray): Текущий кадр изображения.
        fps_counter (FPS_Counter): Объект класса FPS_Counter для вычисления FPS.

    Returns:
        numpy.ndarray: Кадр с отображенным значением FPS.
    """
    fps_real = fps_counter.calc_FPS()
    text = f"FPS: {fps_real:.1f}"

    # Параметры для шрифтов:
    fontFace = 1
    fontScale = 1.3
    thickness = 1
    
    (label_width, label_height), _ = cv2.getTextSize(
        text,
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
    )
    frame = cv2.rectangle(frame, (0, 0), (10 + label_width, 15 + label_height), (0, 0, 0), -1)
    frame = cv2.putText(
        img=frame,
        text=text,
        org=(5, 20),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color=(255, 255, 255),
    )
    return frame

#fps_counter = FPS_Counter(calc_time_perion_N_frames=10) # Определяем размер усреднения

# Define the parameters
imgsz = 640
conf = 0.4
iou = 0.7

# Load the YOLOv8 model
model = YOLO("yolov8m.pt")  
#инициализация веб-камеры
cap = cv2.VideoCapture(0)

#проверка успешности открытия
if not cap.isOpened():
    print("Не удалось открыть веб-камеру")

#установка разрешения (1280*720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    
while True:
    #захват кадра с веб-камеры
    ret, frame = cap.read()
    #frame - np array, bgr цвет

    #проверка успешности открытия
    if not ret:
        print("Не удалось получить кадр с веб-камеры")
        break

    frame = visualize_results_usual_yolo_inference(
        frame,
        model,
        imgsz,
        conf,
        iou,
        segment=False,
        delta_colors=1,
        thickness=6,
        font_scale=1.5,
        show_boxes=True,
        show_confidences=False,
        return_image_array=True,
    )

    # ресайз
    scale = 1
    frame = cv2.resize(frame, (-1, -1), fx=scale, fy=scale)

    #frame = calc_and_show_fps(frame, fps_counter)

    cv2.imshow("WebCam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#освобождение ресрсов
cap.release()
cv2.destroyAllWindows()