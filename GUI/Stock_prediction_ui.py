import sys
import numpy as np
import networkx as nx

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QDialog
from PyQt5.QtGui import QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import Representation_of_Stock_Graph
import Stock_prediction_main
import Prediction_result


class parentWindow(QMainWindow):
  def __init__(self):
    QMainWindow.__init__(self)
    self.main_ui = Stock_prediction_main.Ui_MainWindow()
    self.main_ui.setupUi(self)
class childWindow1(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    self.child=Representation_of_Stock_Graph.Ui_Dialog()
    self.child.setupUi(self)

class childWindow2(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    self.child=Prediction_result.Ui_Dialog()
    self.child.setupUi(self)


if __name__=='__main__':
  app=QApplication(sys.argv)
  window=parentWindow()
  child1=childWindow1()
  child2=childWindow2()
  #通过toolButton将两个窗体关联
  btn1=window.main_ui.pushButton
  btn1.clicked.connect(child1.show)
  btn2=window.main_ui.pushButton_2
  btn2.clicked.connect(child2.show)

  # 显示
  window.show()
  sys.exit(app.exec_())
