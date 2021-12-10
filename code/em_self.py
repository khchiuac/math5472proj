# -----------------------------------------------------------
# self implementation of EM algorithm, without optimization
# 
# December 2021
# -----------------------------------------------------------
from util import *

class EM:
    def __init__(self, L, x_init, update_thres, step_limit):
        self.L = L
        self.n, self.m = L.shape
        self.x_0 = x_init
        self.epsilon = update_thres
        self.step_limit = step_limit

    def em(self):
        cur_x = deepcopy(self.x_0)
        # first step
        update_x = em_update(self.L, cur_x)
        update_x /= np.sum(update_x)

        for _ in range(self.step_limit-1):
            if abs(np.sum(update_x-cur_x))<self.epsilon:
                break
            else:
                cur_x = update_x
                update_x = em_update(self.L, cur_x)
                update_x /= np.sum(update_x)

        return update_x

if __name__ == "__main__":
    n, m = (30, 20)
    thres = 1e-5
    max_steps = 1000
    L = rand(n, m)
    x_0 = dirichlet(np.ones(m), size=1)[0]

    cur_em = EM(L, x_0, thres, max_steps)
    start_t = time.process_time()
    cur_em.em()
    end_t = time.process_time()
    print(end_t-start_t)
