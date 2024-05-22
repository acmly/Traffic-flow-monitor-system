import supervision as sv
from ultralytics import YOLO
from ultralytics.data.loaders import LoadStreams

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import DEFAULT_CFG, SETTINGS
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.files import increment_path
from ultralytics.cfg import get_cfg
from ultralytics.utils.checks import check_imshow

from PySide6.QtCore import Signal, QObject

from pathlib import Path
import datetime
import numpy as np
import time
import cv2

from classes.paint_trail import draw_trail
from utils.main_utils import check_path

x_axis_time_graph = []
y_axis_count_graph = []
video_id_count = 0


class YoloPredictor(BasePredictor, QObject):
    yolo2main_trail_img = Signal(np.ndarray)
    yolo2main_box_img = Signal(np.ndarray)
    yolo2main_status_msg = Signal(str)
    yolo2main_fps = Signal(str)

    yolo2main_labels = Signal(dict)
    yolo2main_progress = Signal(int)
    yolo2main_class_num = Signal(int)
    yolo2main_target_num = Signal(int)


    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        try:
            self.args = get_cfg(cfg, overrides)
        except:
            pass
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        self.used_model_name = None
        self.new_model_name = None

        self.source = ''
        self.progress_value = 0

        self.stop_dtc = False
        self.continue_dtc = True

        self.iou_thres = 0.45
        self.conf_thres = 0.25
        self.speed_thres = 0.01

        self.save_res = False
        self.save_txt = False
        self.save_res_path = "pre_result"
        self.save_txt_path = "pre_labels"

        self.show_labels = True
        self.show_trace = True


        self.start_time = None
        self.count = None
        self.sum_of_count = None
        self.class_num = None
        self.total_frames = None
        self.lock_id = None

        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )

    @smart_inference_mode()
    def run(self):
        self.yolo2main_status_msg.emit('Loading model...')
        LoadStreams.capture = ''
        self.count = 0 
        self.start_time = time.time()
        global video_id_count

        if self.save_txt:
            check_path(self.save_txt_path)
        if self.save_res:
            check_path(self.save_res_path)

        model = self.load_yolo_model()

        iter_model = iter(
            model.track(source=self.source, show=False, stream=True, iou=self.iou_thres, conf=self.conf_thres))

        global x_axis_time_graph, y_axis_count_graph
        x_axis_time_graph = []
        y_axis_count_graph = []

        self.yolo2main_status_msg.emit('Detecting...')

        if 'mp4' in self.source or 'avi' in self.source or 'mkv' in self.source or 'flv' in self.source or 'mov' in self.source:
            cap = cv2.VideoCapture(self.source)
            self.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

        img_res, result, height, width = self.recognize_res(iter_model)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        if self.save_res:
            out = cv2.VideoWriter(f'{self.save_res_path}/video_result_{video_id_count}.mp4', fourcc, 25,
                                  (width, height), True)

        while True:
            try:
                if self.continue_dtc:
                    img_res, result, height, width = self.recognize_res(iter_model)
                    self.res_address(img_res, result, height, width, model, out)

                if self.stop_dtc:
                    if self.save_res:
                        if out:
                            out.release()
                            video_id_count += 1
                    self.source = None
                    self.yolo2main_status_msg.emit('Detection termination')
                    self.release_capture()
                    break


            except StopIteration:
                if self.save_res:
                    out.release()
                    video_id_count += 1
                    print('writing complete')
                self.yolo2main_status_msg.emit('Detection completed')
                self.yolo2main_progress.emit(1000)
                cv2.destroyAllWindows()
                self.source = None

                break
        try:
            out.release()
        except:
            pass

    def res_address(self, img_res, result, height, width, model, out):
            img_box = np.copy(img_res) 
            img_trail = np.copy(img_res)

            if result.boxes.id is None:
                self.sum_of_count = 0
                self.class_num = 0
                labels_write = "No targets identified！"
            else:
                detections = sv.Detections.from_ultralytics(result)
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

                self.class_num = self.get_class_number(detections)
                id = detections.tracker_id
                xyxy = detections.xyxy
                self.sum_of_count = len(id)

                if self.show_trace:
                    img_trail = np.zeros((height, width, 3), dtype='uint8')
                    identities = id
                    grid_color = (255, 255, 255)
                    line_width = 1
                    grid_size = 100
                    for y in range(0, height, grid_size):
                        cv2.line(img_trail, (0, y), (width, y), grid_color, line_width)
                    for x in range(0, width, grid_size):
                        cv2.line(img_trail, (x, 0), (x, height), grid_color, line_width)
                    draw_trail(img_trail, xyxy, model.model.names, id, identities)
                else:
                    img_trail = img_res

                labels_write, img_box = self.creat_labels(detections, img_box , model)


            if self.save_txt:
                with open(f'{self.save_txt_path}/result.txt', 'a', encoding='utf-8') as f:
                    f.write('Current screen information:' +
                            str(labels_write) +
                            f'Detection time: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}' +
                            f'Total number of targets adopted by road sections: {self.sum_of_count}')
                    f.write('\n')


            if self.save_res:
                out.write(img_box)

            now = datetime.datetime.now()
            new_time = now.strftime("%Y-%m-%d %H:%M:%S")
            if new_time not in x_axis_time_graph:
                x_axis_time_graph.append(new_time)
                y_axis_count_graph.append(self.sum_of_count)


            if self.lock_id is not None:
                self.lock_id = int(self.lock_id)
                self.open_target_tracking(detections=detections, img_res=img_res)

            self.emit_res(img_trail, img_box)

    def recognize_res(self, iter_model):
            result = next(iter_model) 
            img_res = result.orig_img
            height, width, _ = img_res.shape

            return img_res, result, height, width

    def open_target_tracking(self, detections, img_res):
        try:
            result_cropped = self.single_object_tracking(detections, img_res)
            cv2.imshow(f'OBJECT-ID:{self.lock_id}', result_cropped)
            cv2.moveWindow(f'OBJECT-ID:{self.lock_id}', 0, 0)
            if cv2.waitKey(5) & 0xFF == 27:
                self.lock_id = None
                cv2.destroyAllWindows()
        except:
            cv2.destroyAllWindows()
            pass

    def single_object_tracking(self, detections, img_box):
        store_xyxy_for_id = {}
        for xyxy, id in zip(detections.xyxy, detections.tracker_id):
            store_xyxy_for_id[id] = xyxy
            mask = np.zeros_like(img_box)
        try:
            if self.lock_id not in detections.tracker_id:
                cv2.destroyAllWindows()
                self.lock_id = None
            x1, y1, x2, y2 = int(store_xyxy_for_id[self.lock_id][0]), int(store_xyxy_for_id[self.lock_id][1]), int(
                store_xyxy_for_id[self.lock_id][2]), int(store_xyxy_for_id[self.lock_id][3])
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            result_mask = cv2.bitwise_and(img_box, mask)
            result_cropped = result_mask[y1:y2, x1:x2]
            result_cropped = cv2.resize(result_cropped, (256, 256))
            return result_cropped

        except:
            cv2.destroyAllWindows()
            pass

    def emit_res(self, img_trail, img_box):

        time.sleep(self.speed_thres/1000) 
        self.yolo2main_trail_img.emit(img_trail)
        self.yolo2main_box_img.emit(img_box)
        self.yolo2main_class_num.emit(self.class_num)
        self.yolo2main_target_num.emit(self.sum_of_count)
        if '0' in self.source or 'rtsp' in self.source:
            self.yolo2main_progress.emit(0)
        else:
            self.progress_value = int(self.count / self.total_frames * 1000)
            self.yolo2main_progress.emit(self.progress_value)
        self.count += 1
        if self.count % 3 == 0 and self.count >= 3:
            self.yolo2main_fps.emit(str(int(3 / (time.time() - self.start_time))))
            self.start_time = time.time()

    def load_yolo_model(self):
        if self.used_model_name != self.new_model_name:
            self.setup_model(self.new_model_name)
            self.used_model_name = self.new_model_name
        return YOLO(self.new_model_name)

    def creat_labels(self, detections, img_box, model):
        xyxy = detections.xyxy
        confidence = detections.confidence
        class_id = detections.class_id
        tracker_id = detections.tracker_id

        labels_draw = [
            f"ID: {tracker_id[i]} {model.model.names[class_id[i]]}"
            for i in range(len(confidence))
        ]
        
        labels_write = [
            f"OBJECT-ID: {tracker_id[i]} CLASS: {model.model.names[class_id[i]]} CF: {confidence[i]:0.2f}"
            for i in range(len(confidence))
        ]

        if (self.show_labels == True) and (self.class_num != 0):
            img_box = self.box_annotator.annotate(scene=img_box, detections=detections, labels=labels_draw)

        return labels_write, img_box



    def get_class_number(self, detections):
        class_num_arr = []
        for each in detections.class_id:
            if each not in class_num_arr:
                class_num_arr.append(each)
        return len(class_num_arr)

    def release_capture(self):
        LoadStreams.capture = 'release'