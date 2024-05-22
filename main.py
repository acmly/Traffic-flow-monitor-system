from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor,QCursor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, Qt

from ui.main_window import Ui_MainWindow
from ui.pop.pop_box import MessageBox
from ui.ui_function import *
from ui.toast.toast import DialogOver
from ui.dialog.rtsp_win import Window
from ui.dialog.id_win import id_Window

from utils.main_utils import check_url, check_path
from utils.AtestCamera import Camera
from classes.yolo import YoloPredictor
from classes.main_config import MainConfig
from classes.car_chart import WorkerThread

from PIL import Image
import numpy as np
import supervision as sv
import subprocess
import sys
import cv2
import os
import datetime


class MainWindow(QMainWindow, Ui_MainWindow):

    # Main window sends execution signal to yolo instance
    main2yolo_begin_sgl = Signal()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        self.setupUi(self)
        # Set background to translucent & frameless window
        self.setAttribute(Qt.WA_TranslucentBackground) 
        self.setWindowFlags(Qt.FramelessWindowHint)

        # UI actions (effects)
        UIFuncitons.uiDefinitions(self)

        # Variable settings
        self.car_threshold = 0 # Vehicle threshold
        self.web_flag = True # Can be opened
        self.server_process = None # Server process
        self.image_id = 0 # Image ID
        self.txt_id = 0 # Label ID

        # Set shadow
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162,129,247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # Set to --
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')

        # Timer: Monitor model file changes every 2 seconds
        self.Qtimer_ModelBox = QTimer(self)
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)


        self.yolo_init()  
        self.model_bind() 
        self.main_function_bind()

        self.load_config() 
        self.model_load()  

        # Draw thread
        self.is_draw_thread = False
        self.draw_thread = WorkerThread()

        self.show_status('Welcome to the Traffic Flow Monitor System')

    def yolo_init(self):
        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()
        self.select_model = self.model_box.currentText()
        self.yolo_thread = QThread()

        # Show prediction video (left, right)
        self.yolo_predict.yolo2main_trail_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_box_img.connect(lambda x: self.show_image(x, self.res_video))

        # Output information, FPS, number of classes, total
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        self.yolo_predict.yolo2main_class_num.connect(lambda x:self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))

        # Move to the thread (controlled by main2yolo_begin_sgl signal - yolo_thread thread must be started first, then yolo_predict's run method can be started)
        self.yolo_predict.moveToThread(self.yolo_thread)
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)

        # Show total traffic flow
        self.yolo_predict.yolo2main_target_num.connect(lambda x:self.Target_setText(x))

    # Load the model
    def model_load(self):
        check_path(self.config.models_path)
        self.model_box.clear()
        self.pt_list = os.listdir(f'./{self.config.models_path}')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt') or file.endswith('.engine')]
        self.pt_list.sort(key=lambda x: os.path.getsize(f'./{self.config.models_path}/' + x))   
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)

    def main_function_bind(self):
        self.src_file_button.clicked.connect(self.open_src_file) 
        self.src_cam_button.clicked.connect(self.camera_select) 
        self.src_rtsp_button.clicked.connect(self.rtsp_seletction) 
        self.src_graph_button.clicked.connect(self.show_traffic_graph) 
        self.src_lock_button.clicked.connect(self.lock_id_selection) 
        self.src_web_button.clicked.connect(self.web_back_end)
        self.run_button.clicked.connect(self.run_or_continue)
        self.stop_button.clicked.connect(self.stop) 

        self.save_res_button.toggled.connect(self.is_save_res)
        self.save_txt_button.toggled.connect(self.is_save_txt)
        self.show_labels_checkbox.toggled.connect(self.is_show_labels) 
        self.show_trace_checkbox.toggled.connect(self.is_show_trace)

        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))

    def model_bind(self):
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'iou_spinbox'))    # iou box
        self.iou_slider.valueChanged.connect(lambda x:self.change_val(x, 'iou_slider'))      # iou scroll bar

        self.conf_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x:self.change_val(x, 'conf_slider'))    # conf scroll bar

        self.speed_spinbox.valueChanged.connect(lambda x:self.change_val(x, 'speed_spinbox'))# speed box
        self.speed_slider.valueChanged.connect(lambda x:self.change_val(x, 'speed_slider'))  # speed scroll bar

        self.speed_sss.valueChanged.connect(lambda x: self.change_val(x, 'speed_sss'))  # speed box
        self.speed_nnn.valueChanged.connect(lambda x:self.change_val(x, 'speed_nnn'))  # speed scroll bar

    def load_config(self):
        self.config = MainConfig("./config/config.json")

        self.save_res_button.setChecked(self.config.save_res)
        self.save_txt_button.setChecked(self.config.save_txt)

        self.iou_slider.setValue(self.config.iou * 100)
        self.conf_slider.setValue(self.config.conf * 100)
        self.speed_slider.setValue(self.config.rate)
        self.speed_sss.setValue(self.config.car_threshold)

        self.yolo_predict.save_txt = self.config.save_txt 
        self.yolo_predict.save_res = self.config.save_res 
        self.yolo_predict.save_txt_path = self.config.save_txt_path
        self.yolo_predict.save_res_path = self.config.save_res_path
        self.yolo_predict.new_model_name = f"./{self.config.models_path}/%s" % self.select_model

        self.yolo_predict.show_trace = self.config.show_trace  
        self.show_trace_checkbox.setChecked(self.config.show_trace)
        self.yolo_predict.show_labels = self.config.show_labels 
        self.show_labels_checkbox.setChecked(self.config.show_labels)

        self.open_fold = self.config.open_fold
        self.rtsp_ip = self.config.rtsp_ip
        self.car_id = self.config.car_id

        self.run_button.setChecked(False)

    def Target_setText(self, num):
        num = str(num)
        self.Target_num.setText(num)
        self.char_label.setText(f"Current traffic Volume: {num}")
        if (int(num) > int(self.car_threshold)):
            self.char_label.setStyleSheet("color: red;")
        else:
            self.char_label.setStyleSheet("color: green;")


# Main window displays trajectory image and detection image (scaling here)
    @staticmethod
    def show_image(img_src, label):
        try:
            # Check the number of channels in the image to determine if it is a color image
            if len(img_src.shape) == 3:
                ih, iw, _ = img_src.shape
            if len(img_src.shape) == 2:
                ih, iw = img_src.shape

            w = label.geometry().width()
            h = label.geometry().height()

            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)

            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def run_or_continue(self):
        if self.yolo_predict.new_model_name == '' or self.yolo_predict.new_model_name == None:
            DialogOver(parent=self, text="Please check models", title="Fail", flags="danger")
            self.run_button.setChecked(False)
            return
        if self.yolo_predict.source == '' or self.yolo_predict.source == None:
            self.show_status('Please choose input before start')
            self.run_button.setChecked(False)
            DialogOver(parent=self, text="Please check input", title="Fail", flags="danger")
            return

        self.yolo_predict.stop_dtc = False 

        if self.run_button.isChecked():

            file_extension = self.yolo_predict.source[-3:].lower()
            if file_extension == 'png' or file_extension == 'jpg':
                self.img_predict()
                return

            DialogOver(parent=self, text="Start monitoring...", title="Success", flags="success")
            self.run_button.setChecked(True)

            self.draw_thread.run_continue() 
            self.save_txt_button.setEnabled(False)
            self.save_res_button.setEnabled(False)
            self.conf_slider.setEnabled(False)
            self.iou_slider.setEnabled(False)
            self.speed_slider.setEnabled(False)

            self.show_status('Monitoring...')
            if '0' in self.yolo_predict.source or 'rtsp' in self.yolo_predict.source:
                self.progress_bar.setFormat('Live stream monitoring...')
            if 'avi' in self.yolo_predict.source or 'mp4' in self.yolo_predict.source:
                self.progress_bar.setFormat("Current monitor progress:%p%")
            self.yolo_predict.continue_dtc = True
            if not self.yolo_thread.isRunning():
                self.yolo_thread.start()
                self.main2yolo_begin_sgl.emit()

        else:
            self.draw_thread.pause() 
            self.yolo_predict.continue_dtc = False
            self.show_status("Pause...")
            DialogOver(parent=self, text="Monitor Paused", title="Pause", flags="warning")
            self.run_button.setChecked(False)

    def show_status(self, msg):
        self.status_bar.setText(msg)  
        if msg == 'Monitor Complete':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()
            self.draw_thread.stop()
            self.is_draw_thread = False


        elif msg == 'Monitor aborted':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)    
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()
            self.draw_thread.stop()
            self.is_draw_thread = False

            self.pre_video.clear()          
            self.res_video.clear()          
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')

    def open_src_file(self):

        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', self.open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('Load file: {}'.format(os.path.basename(name)))
            self.open_fold = os.path.dirname(name)
            self.stop()
            DialogOver(parent=self, text=f"File path: {name}", title="Load success", flags="success")

    def camera_select(self):
        #try:

            self.stop()
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.src_cam_button.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 20px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 212, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(16,155,226,50);
                                            }
                                            ''')
            
            for cam in cams:
                exec("action_%s = QAction('Camera No. %s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)
            pos = QCursor.pos()
            action = popMenu.exec(pos)

            if action:
                str_temp = ''
                selected_stream_source = str_temp.join(filter(str.isdigit, action.text())) 
                self.yolo_predict.source = selected_stream_source
                self.show_status(f'Camera Device:{action.text()}')
                DialogOver(parent=self, text=f"Current Camera: {action.text()}", title="Camera selected", flags="success")

    def rtsp_seletction(self):
        self.rtsp_window = Window()
        self.rtsp_window.rtspEdit.setText(self.rtsp_ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):

        MessageBox(self.close_button, title='Note', text='Load rtsp...', time=1000, auto=True).exec()
        self.stop() 

        self.yolo_predict.source = ip
        self.rtsp_ip = ip 
        self.rtsp_window.close()

        self.show_status(f'Loading rtsp address: {ip}')
        DialogOver(parent=self, text=f"rtsp address is: {ip}", title="rtsp loaded", flags="success")

    def lock_id_selection(self):
        self.yolo_predict.lock_id = None
        self.id_window = id_Window()
        self.id_window.idEdit.setText(str(self.car_id))
        self.id_window.show()
        self.id_window.idButton.clicked.connect(lambda: self.set_lock_id(self.id_window.idEdit.text()))

    def set_lock_id(self,lock_id):
        self.yolo_predict.lock_id = None
        self.yolo_predict.lock_id = lock_id
        self.car_id = lock_id  
        self.show_status('load ID:{}'.format(lock_id))
        self.id_window.close()

    def show_traffic_graph(self):
        if not self.run_button.isChecked():
            DialogOver(parent=self, text="Please start monitoring", title="Failed", flags="danger")
            return

        if self.is_draw_thread:
            DialogOver(parent=self, text="Flow chart initicated", title="Already started", flags="danger")
            return
        self.draw_thread.start()
        self.is_draw_thread = True

    def web_back_end(self):
        self.stop()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        flask_app_path = os.path.join(base_dir, 'app.py')

        if (self.web_flag):
            self.server_process = subprocess.Popen(['python', flask_app_path])
            MessageBox(self.close_button, title='Note', text='Starting website...', time=2000, auto=True).exec()

            try:
                if self.server_process.pid is not None:
                    self.src_web_button.setText("Close website")
                    self.web_flag = False
                    DialogOver(parent=self, text="Website started", title="Start", flags="success")
            except Exception as e:
                DialogOver(parent=self, text=str(e), title="Fail", flags="danger")

        else:
            try:
                self.server_process.terminate()
                MessageBox(self.close_button, title='Note', text='Closing website...', time=2000, auto=True).exec()

                self.src_web_button.setText("Website started")
                self.web_flag = True
                DialogOver(parent=self, text="Close out Website", title="Closed", flags="success")
            except Exception as e:
                DialogOver(parent=self, text=str(e), title="Closing failed", flags="danger")

    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('Note: Monitoring results will not be saved.')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('Note: Monitoring results will be saved.')
            self.yolo_predict.save_res = True
    
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('Note: labels will not be saved.')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('Note: labels will be saved.')
            self.yolo_predict.save_txt = True

    def is_show_labels(self):
        if self.show_labels_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.yolo_predict.show_labels = False
            self.show_status('Note:labels off')
        elif self.show_labels_checkbox.checkState() == Qt.CheckState.Checked:
            self.yolo_predict.show_labels = True
            self.show_status('Note:labels on')

    def is_show_trace(self):
        if self.show_trace_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.yolo_predict.show_trace = False
            self.show_status('Note: traces off')
        elif self.show_trace_checkbox.checkState() == Qt.CheckState.Checked:
            self.yolo_predict.show_trace = True
            self.show_status('Note: traces on')

    def stop(self):
        try:
            self.yolo_predict.release_capture()  
            self.yolo_thread.quit()

        except:
            pass
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)   

        self.save_res_button.setEnabled(True)   
        self.save_txt_button.setEnabled(True)  
        self.iou_slider.setEnabled(True)        
        self.conf_slider.setEnabled(True)       
        self.speed_slider.setEnabled(True)     
        self.pre_video.clear() 
        self.res_video.clear() 
        self.progress_bar.setValue(0) 
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')


    def change_val(self, x, flag):

        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x*100))    
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x/100) 
            self.show_status('IOU Threshold: %s' % str(x/100))
            self.yolo_predict.iou_thres = x/100

        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x*100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x/100)
            self.show_status('Conf Threshold: %s' % str(x/100))
            self.yolo_predict.conf_thres = x/100

        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms

        elif flag == 'speed_nnn':
            self.speed_sss.setValue(x)
        elif flag == 'speed_sss':
            self.speed_nnn.setValue(x)
            self.show_status('Traffic volume threshold: %s cars' % str(x))
            self.car_threshold = x  # ms

    def change_model(self,x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = f"./{self.config.models_path}/%s" % self.select_model
        self.show_status('Change model:%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    def ModelBoxRefre(self):
        pt_list = os.listdir(f'./{self.config.models_path}')
        pt_list = [file for file in pt_list if file.endswith('.pt') or file.endswith('.engine')]
        pt_list.sort(key=lambda x: os.path.getsize(f'./{self.config.models_path}/' + x))
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    def closeEvent(self, event):
        try:
            self.stop()
            self.draw_thread.close_exec()
            # self.draw_thread.deleteLater()

            # config.json
            self.config.save_res = self.yolo_predict.save_res
            self.config.save_txt = self.yolo_predict.save_txt

            self.config.show_labels = self.yolo_predict.show_labels
            self.config.show_trace = self.yolo_predict.show_trace

            self.config.iou = self.yolo_predict.iou_thres
            self.config.conf = self.yolo_predict.conf_thres
            self.config.rate = self.yolo_predict.speed_thres

            self.config.car_threshold = self.car_threshold  
            self.config.rtsp_ip = self.rtsp_ip
            self.config.car_id = self.car_id
            self.config.open_fold = self.open_fold

            self.config.save_config() 

            MessageBox(
                self.close_button, title='Note', text='Exiting, please wait...', time=2000, auto=True).exec()

            if self.server_process is not None:
                if self.server_process.pid is not None:
                    self.server_process.terminate() 
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(0)

    def img_predict(self):

        if check_url(self.yolo_predict.source):
            DialogOver(parent=self, text="cannot read input path", title="Process abort", flags="danger")
            return

        self.run_button.setChecked(False)  
        image = cv2.imread(self.yolo_predict.source)
        org_img = image.copy()

        model = self.yolo_predict.load_yolo_model()
        iter_model = iter(model.track(source=image, show=False))
        result = next(iter_model)  
        if result.boxes.id is None:
            DialogOver(parent=self, text="No selected object in the image", title="Finish", flags="warning")
            self.show_image(image, self.pre_video)
            self.show_image(image, self.res_video)
            self.yolo_predict.source = ''
            return

        detections = sv.Detections.from_ultralytics(result)
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        labels_write, img_box = self.yolo_predict.creat_labels(detections, image, model)

        self.Class_num.setText(str(self.yolo_predict.get_class_number(detections)))
        self.Target_num.setText(str(len(detections.tracker_id)))
        self.show_image(org_img, self.pre_video)  # left
        self.show_image(img_box, self.res_video)  # right
        self.yolo_predict.source = ''
        DialogOver(parent=self, text="Image detect done", title="Finish", flags="success")


        if self.yolo_predict.save_res:
            check_path(self.config.save_res_path)
            while os.path.exists(f"{self.config.save_res_path}/image_result_{self.image_id}.jpg"):
                self.image_id += 1
            rgb_frame = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
            numpy_frame = np.array(rgb_frame)
            Image.fromarray(numpy_frame).save(f"./{self.config.save_res_path}/image_result_{self.image_id}.jpg")


        if self.yolo_predict.save_txt:
            check_path(self.config.save_txt_path)

            while os.path.exists(f"{self.config.save_txt_path}/result_{self.txt_id}.jpg"):
                self.txt_id += 1

            with open(f'{self.config.save_txt_path}/result_{self.txt_id}.txt', 'a') as f:
                f.write('Current screen information' +
                        str(labels_write) +
                        f'Monitor time: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}' +
                        f' Total number: {len(detections.tracker_id)}')
                f.write('\n')
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())  
