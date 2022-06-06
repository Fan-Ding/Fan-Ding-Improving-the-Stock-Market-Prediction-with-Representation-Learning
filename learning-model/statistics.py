import numpy as np
list=["trained-result-group1-test","trained-result-group2-test","trained-result-group3-test"]
for n in range(1):

    average = 0
    sum = 0
    low=1
    high=0
    f = open(list[0])  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    i=0;
    data=np.zeros((50,2),dtype=float)
    while line:

        # print(line)
        accuracy = line.split()[4]
        accuracy = float(accuracy)
        # print(accuracy)
        if(accuracy > data[i][0]):
            data[i][0] = accuracy
        # if accuracy >= 0.5:
        #     count0 = count0 + 1
        # if accuracy >=0.55:
        #     count1 = count1 + 1
        line = f.readline()
        i=i+1
        i = i % 50
        # if accuracy < low:
        #     low =accuracy
        # if accuracy > high:
        #     high = accuracy

    f.close()
    for i in range(50):
        print(i,data[i][0])
        sum = sum + data[i][0]

    average = sum / 50

    print(average)
