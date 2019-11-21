import json
from itertools import accumulate

from matplotlib import pyplot as plt

# grinch graft analysis
with open("experiment/graft_significance/grinch.json") as file:
    s = file.readlines()
    s = "".join(s)
    data = json.loads(s)
    x = [i + 1 for i in range(len(data))]
    y_instant = list(map(lambda item: item[1] - item[0], data))
    y_accumu = list(accumulate(y_instant, lambda a, b: a + b))
    print(data)
    print(y_instant)
    print(y_accumu)
    plt.plot(x, y_instant, 'r--', label='instantaneous')
    plt.plot(x, y_accumu, 'b--', label='accumulate')
    plt.xlabel('Data Points')  # X轴标签
    plt.ylabel("Dendrogram Purity Change")  # Y轴标签
    plt.legend()
    plt.savefig("/home/troublor/Desktop/1.png")
    plt.show()

    # joint analysis
    with open("experiment/graft_significance/rotation.json") as file1:
        s = file1.readlines()
        s = "".join(s)
        data1 = json.loads(s)
        y_grinch = list(map(lambda item: item[1], data))
        y_rotation = list(map(lambda item: item[1], data1))
        plt.plot(x, y_grinch, 'r--', label='grinch')
        plt.plot(x, y_rotation, 'b--', label='rotation')
        plt.xlabel('Data Points')  # X轴标签
        plt.ylabel("Dendrogram Purity")  # Y轴标签
        plt.legend()
        plt.savefig("/home/troublor/Desktop/2.png")
        plt.show()

