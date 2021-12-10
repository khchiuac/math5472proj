# -----------------------------------------------------------
# files for drawing figures as shown in the report
#
# December 2021
# -----------------------------------------------------------
from util import *
from em_self import *
from ip_self import *
from sqp_self import *

thres = 1e-5
max_steps = 1000
length = 1000000
mu_list = rand(length)
sigma_list = rand(length)

method_list = ["em", "ip", "sqp"]

class GMM:
    def __init__(self, n, m=40):
        self.n = n
        self.m = m
        self.pi = dirichlet(np.ones(m),size=1)[0]
        self.z, self.x_id = make_blobs(n_samples=int(n), centers=mu_list[:m].reshape(-1,1), cluster_std=sigma_list[:m], random_state=0)
        self.x = np.eye(self.m)[self.x_id]
        self.L = np.zeros((n, m))
        for j in range(n):
            for k in range(m):
                self.L[j, k] = norm(mu_list[k], sigma_list[k]).pdf(self.z[j])
    
    def solve_and_time(self, method="em"):
        x_0 = dirichlet(np.ones(self.m), size=1)[0]
        if method=="em":
            cur_em = EM(self.L, x_0, thres, max_steps)
            start_t = time.process_time()
            cur_em.em()
            end_t = time.process_time()
            #print(f"Time for method {method} with n={self.n}, m={self.m}: {end_t-start_t}")
            return end_t-start_t
        elif method=="ip":
            cur_ip = IP(self.L)
            start_t = time.process_time()
            cur_ip.ip()
            end_t = time.process_time()
            return end_t-start_t
        elif method=="sqp":
            cur_sqp = SQP(self.L, x_0, max_steps)
            start_t = time.process_time()
            cur_sqp.sqp()
            end_t = time.process_time()
            return end_t-start_t

if __name__ == "__main__":
    m = 40
    n_list = list(m*2**x for x in range(1,9))
    result_dict = {"em":[], "ip":[], "sqp":[]}
    for method in method_list:
        for n in n_list:
            gmm = GMM(n, m)
            t = gmm.solve_and_time(method)
            print(t)
            result_dict[method].append(t)
        print(result_dict)
    
    with open('../result.json', 'w') as fp:
        json.dump(result_dict, fp)
