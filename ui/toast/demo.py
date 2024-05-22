# -*- coding: utf-8 -*-
# @Author : pan
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PySide6.QtGui import QPixmap, QPainter, QColor, QFontMetrics
from PySide6.QtWidgets import QApplication, QWidget, QLabel

class Toast(QWidget):
    def __init__(
        self,
        text: str,
        duration: int = 3000,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.duration = duration

        label = QLabel(self)
        label.setText(text)
        label.setStyleSheet("""
            background-color: rgba(60, 179, 113, 0.8);
            color: white;
            font-size: 16px;
            padding: 12px;
            border-radius: 4px;
        """)
        label.setAlignment(Qt.AlignCenter)

        fm = QFontMetrics(label.font())
        width = fm.boundingRect(text).width() + 80


        label.setFixedWidth(width)
        label.setFixedHeight(40)

        self.setGeometry(*self.calculatePosition(label.sizeHint()))

        self.fadeIn()

        self.animationTimer = QTimer()
        self.animationTimer.setSingleShot(True)
        self.animationTimer.timeout.connect(self.fadeOut)
        self.animationTimer.start(self.duration)



    def fadeIn(self):

        fadeInAnimation = QPropertyAnimation(self, b"windowOpacity", self)
        fadeInAnimation.setStartValue(0)
        fadeInAnimation.setEndValue(1)
        fadeInAnimation.setDuration(500)
        fadeInAnimation.finished.connect(lambda: print('加载成功'))

        fadeInAnimation.start()

    def calculatePosition(self, sizeHint):
        desktopRect = QApplication.primaryScreen().availableGeometry()
        x = (desktopRect.width() - sizeHint.width()) // 2
        y = desktopRect.height() - sizeHint.height() - 50
        return x, y, sizeHint.width(), sizeHint.height()




    def fadeOut(self):

        self.animationTimer.stop()
        self.animationTimer.timeout.disconnect(self.fadeOut)

        parallelAnimation = QParallelAnimationGroup()

        opacityAnimation = QPropertyAnimation(self, b"windowOpacity")
        opacityAnimation.setStartValue(1.0)
        opacityAnimation.setEndValue(0.0)
        opacityAnimation.setDuration(500)

        yAnimation = QPropertyAnimation(self, b"geometry")
        targetY = self.y() - 50
        yAnimation.setStartValue(self.geometry())
        yAnimation.setEndValue(QApplication.primaryScreen().availableGeometry().translated(0, targetY))
        yAnimation.setDuration(500)
        yAnimation.setEasingCurve(QEasingCurve.OutCubic)

        parallelAnimation.addAnimation(opacityAnimation)
        parallelAnimation.addAnimation(yAnimation)



        parallelAnimation.finished.connect(self.close)

        parallelAnimation.start() 





    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

    def mousePressEvent(self, event):
        pass


if __name__ == "__main__":
    app = QApplication([])

    toast = Toast("Success")
    toast.show()

    app.exec()
