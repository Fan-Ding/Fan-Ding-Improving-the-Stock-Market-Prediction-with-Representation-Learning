#
#
# from tkinter import *  # 导入 Tkinter 库
#
# root = Tk()  # 创建窗口对象的背景色
# # 创建两个列表
# li = ['C', 'python', 'php', 'html', 'SQL', 'java']
# movie = ['CSS', 'jQuery', 'Bootstrap']
# listb = Listbox(root)  # 创建两个列表组件
# listb2 = Listbox(root)
# for item in li:  # 第一个小部件插入数据
#     listb.insert(0, item)
#
# for item in movie:  # 第二个小部件插入数据
#     listb2.insert(0, item)
#
# listb.pack()  # 将小部件放置到主窗口中
# listb2.pack()
# root.mainloop()  # 进入消息循环


import tkinter as Tk

import matplotlib

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure

def _quit():

    root.quit()

    root.destroy()

if __name__ == '__main__':

    root = Tk.Tk()

    root.title("embed matplotlib in TK")

    matplotlib.use('TkAgg')

    fig = Figure(figsize=(5, 5), dpi=100)

    axes = fig.add_subplot(1, 1, 1)

    axes.plot(range(10), range(10)) #绘制图形

    #把绘制的图形显示到tkinter窗口上

    canvas = FigureCanvasTkAgg(fig, master=root)

    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    #把matplotlib绘制图形的导航工具栏显示到tkinter窗口上

    toolbar = NavigationToolbar2Tk(canvas, root)

    toolbar.update()

    button = Tk.Button(master=root, text='Quit', command=_quit)

    button.pack(side=Tk.RIGHT)

    root.mainloop()
