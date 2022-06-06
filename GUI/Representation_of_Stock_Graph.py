# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Representation_of_Stock_Graph.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QInputDialog, QSizePolicy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(789, 613)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(210, 10, 341, 81))
        self.groupBox.setFlat(True)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(180, 40, 121, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(30, 40, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.showDialog)
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(50, 110, 661, 461))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.m = PlotCanvas(self.frame, width=7, height=4)  # 调用PlotCanvas()函数
        self.m.move(0, 0)


        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Input: \'all\' or num in [0,50)"))
        self.pushButton.setText(_translate("Dialog", "input"))

    def showDialog(self):
        text, ok = QInputDialog.getText(self.groupBox, 'Input Dialog',
                                        'Enter stock index:')
        if ok:
            self.lineEdit.setText(str(text))
            self.m.figure.clf()
            self.m.axes = self.m.fig.add_subplot(111)
            self.m.plot(text)
            self.m.figure.canvas.draw()


class PlotCanvas(
    FigureCanvas):  # 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键

    def __init__(self, parent=None, width=100, height=100, dpi=100):
        self.fig = Figure(figsize=(width, height),
                     dpi=dpi)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
        self.axes = self.fig.add_subplot(111)  # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        self.fig.clf()


        FigureCanvas.__init__(self, self.fig)  # 初始化父类
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)



    def plot(self,num):
        G = read_graph(num)

        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.2]

        pos = nx.spring_layout(G)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, ax = self.axes)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, ax = self.axes)
        # nx.draw_networkx_edges(G, pos, edgelist=esmall,width=6, alpha=0.5, edge_color='b', style='dashed')

        # labels,node 的名字
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif',ax = self.axes)

    def plot2(self):
        G = nx.DiGraph()
        G.add_node('555')
        G.add_node('123456')
        G.add_edge('555', '123456')
        nx.draw(G, pos=nx.spring_layout(G), node_color='w', ax=self.axes,
                edge_color='b', with_labels=True, alpha=1,
                font_size=10, node_size=20, arrows=True)


def read_graph(num):

    G = nx.read_edgelist('/Users/dingfan/Desktop/PyCharmProject/GUI/each_stock_graph/'+str(num)+'.edgelist', nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    G = G.to_undirected()

    return G