import matplotlib.pyplot as plt
from util import *

method_list = ["em", "ip", "sqp"]

if __name__ == "__main__":
    f = open('../result.json')
    data = json.load(f)
    plt.figure()
    m=40
    x = list(m*2**x for x in range(1,9))
    for method in method_list:
        plt.plot(x,data[method])
    plt.title("Comparion of problem formulations")
    plt.xlabel("number or rows (n)")
    plt.ylabel("runtime (seconds)")
    plt.legend(labels=method_list)
    plt.savefig('../result.png')
    plt.show()