# -----------------------------------------------------------
# self implementation of MIX-SQP algorithm, without optimization
# 
# December 2021
# -----------------------------------------------------------
from util import *

class SQP:
    def __init__(self, L, x_init, step_limit, method="RRQR", k=2, xi=9.01, rho=0.5, epsilon0=1e-8, epsilon1=1e-10):
        self.L = L
        self.n, self.m = L.shape
        self.x_0 = x_init
        self.step_limit = step_limit
        self.method = method
        self.k = k # for tSVD
        self.xi = xi
        self.rho = rho
        self.epsilon0 = epsilon0
        self.epsilon1 = epsilon1

    def sqp(self):
        if self.method=="RRQR":
            Q, R, P = qr(self.L, mode='full', pivoting=True)
            P = np.diag(P)
        elif self.method=="tSVD":
            U, S, Vt = svds(L, k=k)

        cur_x = deepcopy(self.x_0)
        for t in range(int(self.step_limit)-1):
            d = np.reciprocal(np.matmul(self.L, self.x_0))
            if self.method=="RRQR":
                g = np.transpose(P)@np.transpose(R)@np.transpose(Q)@d/self.n+1
                H = np.transpose(P)@np.transpose(R)@np.transpose(Q)@np.diag(d)@np.diag(d)@Q@R@P/self.n
            elif self.method=="tSVD":
                g = np.transpose(U)@np.transpose(np.diag(S))@np.transpose(Vt)@d/self.n+1
                H = np.transpose(U)@np.transpose(np.diag(S))@np.transpose(Vt)@np.diag(d)@np.diag(d)@Vt@np.diag(S)@U/self.n
            else:
                g = np.transpose(L)@d/self.n+1
                H = np.transpose(L)@np.diag(d)@np.diag(d)@L/self.n
            
            W = set(list(range(1, self.m+1)))
            for i in range(cur_x.shape[0]):
                if cur_x[i]>0:
                    W -= set({i+1})

            y = active_set(g, H, W, self.epsilon1)
            p_t = [y[i]-cur_x[i] for i in range(y.shape[0])]
            if min(g)>=-self.epsilon0:
                break
            alpha_t=1
            while tilde_f(cur_x+alpha_t*p_t)>tilde_f(cur_x)+xi*alpha_t*np.transpose(g)@p:
                alpha_t *= rho
            cur_x += alpha_t*p_t

if __name__ == "__main__":
    n, m = (30, 20)
    max_steps = 1000
    L = rand(n, m)
    x_0 = dirichlet(np.ones(m), size=1)[0]

    cur_sqp = SQP(L, x_0, max_steps)
    start_t = time.process_time()
    cur_sqp.sqp()
    end_t = time.process_time()
    print(end_t-start_t)