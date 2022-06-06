# -*- coding: utf-8 -*-
__author__ = 'djstava@gmail.com'

import sys

from PyQt5.QtWidgets import QApplication , QMainWindow

import firstPyQt5

if __name__ == '__main__':
    '''
    主函数
    '''

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = firstPyQt5.Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())