# Robot Intelligence 2020/12/22
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def Search(S0, G, Ops, QueueFunc, *args, **kwargs):
    """Graph Search
    Parameters
    ----------
    S0 : initial state
    G  : goal state
    Ops: list of operators
    QueueFunc: -
    """
    OL = [S0] # open list
    CL = []   # close list
    Parent = {}

    while OL:
        N = OL.pop(0)
        if N not in CL:
            if N == G:
                return Path(S0, G, Parent), CL, True
            OL = QueueFunc(Expand(N, Ops, Parent, *args, **kwargs), OL)
            CL.append(N)
            
    return None, CL, False

def Expand(N, Ops, Parent, OK=lambda N: True):
    ret = []
    for Op in Ops:
        if OK(Op(N)):
            ret.append(Op(N))
            Parent.setdefault(Op(N), N)
    return ret

def AppendToLast(new, old):
    return old + [e for e in new if e not in old]

def InsertToHead(new, old):
    return [e for e in new if e not in old] + old

def SortByEvalFunc(new, old):
    return sorted(new+old, key=EvalFunc)

def BreadthFirst(S0, G, Ops, *args, **kwargs):
    return Search(S0, G, Ops, AppendToLast, *args, **kwargs)

def DepthFirst(S0, G, Ops, *args, **kwargs):
    return Search(S0, G, Ops, InsertToHead, *args, **kwargs)

def BestFirst(S0, G, Ops, *args, **kwargs):
    return Search(S0, G, Ops, SortByEvalFunc, *args, **kwargs)
    
def Path(S0, G, Parent):
    path = []
    N = G
    while N != S0:
        path.append(N)
        N = Parent[N]
    path.append(S0)
    path.reverse()
    return path


## path searching
# problem setteing
Ops = [lambda N: (N[0]  , N[1]+1),
       lambda N: (N[0]+1, N[1]+1),
       lambda N: (N[0]+1, N[1]  ),
       lambda N: (N[0]+1, N[1]-1),
       lambda N: (N[0]  , N[1]-1),
       lambda N: (N[0]-1, N[1]-1),
       lambda N: (N[0]-1, N[1]  ),
       lambda N: (N[0]-1, N[1]+1)]
obstacle = [(5, j) for j in range(2, 6)]
OK = lambda N: (1<=N[0]<=7) and (1<=N[1]<=5) and (N not in obstacle)
S0 = (2, 3)
G = (7, 3)
EvalFunc = lambda N: (N[0] - G[0])**2 + (N[1] - G[1])**2
# solve the problem
br = BreadthFirst(S0, G, Ops, OK=OK)
dpt = DepthFirst(S0, G, Ops, OK=OK)
bst = BestFirst(S0, G, Ops, OK=OK)

# visualization
def visualize(res, ax1, ax2, *args, **kwargs):
    ax1.grid(c="k")
    ax1.set(aspect="equal", xlim=(0, 7), ylim=(0, 5))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.grid(c="k")
    ax2.set(aspect="equal", xlim=(0, 7), ylim=(0, 5))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    path = np.array(res[0]) - 0.5
    visit = np.array(res[1]) - 0.5
    ax1.plot(path.T[0], path.T[1], *args, **kwargs)
    ax2.plot(visit.T[0], visit.T[1], *args, **kwargs)

plt.close("all")
fig, axes = plt.subplots(1, 2)

for res, method in zip([br, dpt, bst], ["BreathFirst", "DepthFirst", "BestFirst"]):
    print(f"{method}: took {len(res[1])} steps to find the path (length:{len(res[0])})")
    visualize(res, axes[0], axes[1], label=method)

fig.legend()
