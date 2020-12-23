import numpy as np
from scipy.optimize import linprog
import os

os.system('echo \"-inf\" > z_star.dat')


def BB(c, A, b, cur_x=[]):
    """以下の0-1整数計画問題に対する分枝限定法.
    maximize    c.dot(x) 
    subject to  A@x <= b
                and 
                x[i] == 0 or 1 for all i
    """
    z_star = float(open("z_star.dat").readlines()[-1])
    
    # LP緩和を解く
    print(f"c={c}, A={A}, b={b}, z_star={z_star}, cur_x={cur_x}")
    res_LP = linprog(-c, A_ub=A, b_ub=b, bounds=(0.0, 1.0))
    z, x = -res_LP.fun if res_LP.success else -np.inf, res_LP.x
    print(f"z={z}, x={x}")
    # 限定操作を行うか?
    bounded, z_star = bound(z_star, z, x)
    open("z_star.dat", "a").write(f"{z_star}\n")
    if bounded:
        return

    # 分枝操作
    c, A, b1, b2 = branch(c, A, b)
    BB(c, A, b1, [1]+cur_x)
    BB(c, A, b2, [0]+cur_x)
    
    
def branch(c, A, b):
    return c[1:], A[:, 1:], (b - A[:, 0]), b
    
def bound(z_star, z, x):
    if z == -np.inf:
        print("case1")
        return True, z_star
    if z <= z_star:
        print("case2")
        return True, z_star
    if is_zero_one(x):
        if z > z_star:
            print("case3-1")
            return True, z
        else:
            print("case3-2")
            return True, z_star
    return False, z_star

def is_zero_one(x):
    for e in x:
        if (not np.isclose(e, 0.0)) and (not np.isclose(e, 1.0)):
            return False
    return True
        
