import time

from PySide6.QtCore import QThread, Signal

import matplotlib.pyplot as plt
import mplcyberpunk
import matplotlib
matplotlib.use('TkAgg')


class WorkerThread(QThread):

    def __init__(self):
        super().__init__()
        self.is_stopped = True
        self.is_continue = True
        self.is_close = True
        self.is_exec = True

    def run(self):
        self.is_stopped = False
        self.is_continue = False
        self.is_close = False
        self.is_exec = False

        plt.style.use("cyberpunk")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['toolbar'] = 'None'
        plt.figure("MTSP system chart")

        fig = plt.gcf()
        fig.canvas.mpl_connect("close_event", self.on_close)
        while True:

            if self.is_stopped:
                plt.show()
                break

            if self.is_continue:
                time.sleep(1)
                continue

            if self.is_close:
                return
            plt.cla()
            from classes.yolo import y_axis_count_graph as y
            plt.xlabel('Time')
            plt.ylabel('Traffic volume/vehicle')
            plt.title('Real-time flow line graph')
            plt.plot(y, linestyle='-', marker='o')
            mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5, gradient_start='zero')
            plt.xticks([])
            plt.pause(2)


    def on_close(self, event):
        self.is_close = True

    def stop(self):
        self.is_stopped = True

    def pause(self):
        self.is_continue = True

    def run_continue(self):
        self.is_continue = False

    def close_exec(self):
        try:
            self.stop()
            plt.close()
        except Exception as e:
            print(e)
            pass


