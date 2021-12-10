# -----------------------------------------------------------
# self implementation of IP algorithm, without optimization
# 
# December 2021
# -----------------------------------------------------------
from util import *

class IP:
    def __init__(self, L):
        self.L = L
        self.n, self.m = L.shape

    def ip(self):
        x = cp.Variable(self.m)
        objective = cp.Maximize(cp.sum(cp.log(self.L @ x)))
        constraints = [i<=self.n for i in self.L.T@(self.L@x)]
        constraints += [j>=0 for j in self.L@x]
        constraints += [cp.sum(x)==1]
        constraints += [i>=0 for i in x]
        prob = cp.Problem(objective, constraints)

        result = prob.solve()
        return x.value

if __name__ == "__main__":
    n, m = (30, 20)
    L = rand(n, m)

    cur_ip = IP(L)
    start_t = time.process_time()
    cur_ip.ip()
    end_t = time.process_time()
    print(end_t-start_t)
