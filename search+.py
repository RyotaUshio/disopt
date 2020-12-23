# Robot Intelligence 2020/12/22
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

    def has_no_parent(self):
        return not isinstance(self.parent, Node)

    def is_root(self):
        return self.has_no_parent()

    def is_goal(self):
        return self.goal

    def g(self):
        if self.is_root():
            return 0
        return self.parent.g() + 1

    def __eq__(self, rhs):
        return (self.state == rhs.state)



def Search(S0, G, Ops, QueueFunc, *args, **kwargs):
    """Graph Search
    Parameters
    ----------
    S0 : initial node
    G  : goal node
    Ops: list of operators
    QueueFunc: -
    """
    OL = [S0]
    CL = []  
    
    while OL:
        N = OL.pop(0)
        if N not in CL:
            if N == G:
                return (Path(S0, N), CL)
            OL = QueueFunc(Expand(N, Ops, *args, **kwargs), OL)
            CL.append(N)
            
    return (None, CL)

def Expand(N, Ops):
    ret = []
    for Op in Ops:
        try:
            child = Op(N)
        except:
            continue

        if child.has_no_parent():
            child.parent = N
        ret.append(child)
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
    
def Path(S0, G):
    path = []
    N = G
    while not N.is_root():
        path.append(N)
        N = N.parent
    path.append(S0)
    path.reverse()
    return path


### 8 puzzle
class EightPuzzleNode(Node):
    def __eq__(self, rhs):
        return (self.state == rhs.state).all()

    def __repr__(self):
        tmp = self.state.reshape((3, 3))
        rep = ""
        for i in range(3):
            for j in range(3):
                num = tmp[i,j]
                if num:
                    rep += (str(num) + " ")
                else:
                    rep += "  "
            rep += "\n"
        return rep

    def where_space(self):
        [x], [y] = np.where(self.state == 0)
        return (x, y)

    def move(self, loc):
        if not self.movable(loc):
            raise Exception("unmovable")
        tmp = self.copy()
        tmp.state[self.where_space()] = self.state[loc]
        tmp.state[loc] = 0
        return tmp

    def movable(self, loc):
        return (self.is_inside(loc) and
                (np.linalg.norm(
                    np.array(self.where_space()) - np.array(loc), ord=1)
                 == 1))

    def copy(self):
        return EightPuzzleNode(state=self.state.copy())
    # DONT DO ..., parent=self.parent) !!

    def move_space(self, dir):
        space = self.where_space()
        loc = tuple(np.array(space) + np.array(dir))
        return self.move(loc)
        

    @staticmethod
    def is_inside(loc):
        return 0 <= loc[0] <= 2 and 0 <= loc[1] <= 2
        


def h(x, y):
    return len(x.state[x.state != y.state])

S0 = EightPuzzleNode(np.array([[0, 4, 2], [6, 3, 8], [1, 7, 5]]))
G = EightPuzzleNode(np.array([[4, 3, 2], [6, 7, 8], [1, 5, 0]]))
Ops = [lambda N: N.move_space((1,0)),
       lambda N: N.move_space((0,1)),
       lambda N: N.move_space((-1,0)),
       lambda N: N.move_space((0,-1))]
EvalFunc = lambda N: N.g() + h(N, G)
sol = BestFirst(S0, G, Ops)


# ## path searching
# # problem setteing
# Ops = [lambda N: Node((N[0]  , N[1]+1)),
#        lambda N: Node((N[0]+1, N[1]+1)),
#        lambda N: Node((N[0]+1, N[1]  )),
#        lambda N: Node((N[0]+1, N[1]-1)),
#        lambda N: Node((N[0]  , N[1]-1)),
#        lambda N: Node((N[0]-1, N[1]-1)),
#        lambda N: Node((N[0]-1, N[1]  )),
#        lambda N: Node((N[0]-1, N[1]+1))]
# obstacle = [(5, j) for j in range(2, 6)]
# OK = lambda N: (1<=N[0]<=7) and (1<=N[1]<=5) and (N not in obstacle)
# S0 = Node((2, 3))
# G = Node((7, 3), goal=True)
# EvalFunc = lambda N: (N[0] - G[0])**2 + (N[1] - G[1])**2 + N.g()
# # solve the problem
# br = BreadthFirst(S0, G, Ops, OK=OK)
# dpt = DepthFirst(S0, G, Ops, OK=OK)
# bst = BestFirst(S0, G, Ops, OK=OK)

# # visualization
# def visualize(res, ax1, ax2, *args, **kwargs):
#     ax1.grid(c="k")
#     ax1.set(aspect="equal", xlim=(0, 7), ylim=(0, 5))
#     ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax2.grid(c="k")
#     ax2.set(aspect="equal", xlim=(0, 7), ylim=(0, 5))
#     ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
#     path = np.array(res[0]) - 0.5
#     visit = np.array(res[1]) - 0.5
#     ax1.plot(path.T[0], path.T[1], *args, **kwargs)
#     ax2.plot(visit.T[0], visit.T[1], *args, **kwargs)

# plt.close("all")
# fig, axes = plt.subplots(1, 2)

# for res, method in zip([br, dpt, bst], ["BreathFirst", "DepthFirst", "BestFirst"]):
#     print(f"{method}: took {len(res[1])} steps to find the path (length:{len(res[0])})")
#     visualize(res, axes[0], axes[1], label=method)

# fig.legend()
