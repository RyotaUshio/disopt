# 最適化手法 2020/12/21 最短路問題
import numpy as np

class Node:
    def __init__(self, state, adj=None, prev=None):
        self.state = state
        self.adj = list() if adj is None else adj
        self.prev = prev
        self.d = np.inf
        self.c = "unvisited"
        self.opt = [np.inf]

    def __repr__(self):
        return f"Node(state={self.state}, d={self.d}, c={self.c}, opt[-1]={self.opt[-1]})"

    def connect(self, nodes, ls=None):
        """
        nodes: array-like of nodes that are connected to self
        """
        if ls is None:
            ls = [1 for _ in range(len(nodes))]
        for n, l in zip(nodes, ls):
            self.adj.append(Edge((self, n), l))

class Edge:
    def __init__(self, nodes, l=1):
        """
        nodes: a pair of (initial and terminal) vertices of the edge
        l    : length (weight) of the edge
        """
        self.nodes = nodes
        self.l = l

    def __repr__(self):
        return "Edge(" + repr(tuple(self.nodes)) + ")"

def Reset(V):
    for v in V:
        v.d, v.c = np.inf, "unvisited"

# 連結性判定(最短路問題の実行可能性判定)アルゴリズム
def BreadthFirst(V, s=0, n=0):
    """
    V: list of Nodes
    s: index of the initial node
    """
    V[s].d = 0
    Q = [V[s]]

    while Q:
        v = Q.pop(n)
        print(v.state)
        for edge in v.adj:
            w = edge.nodes[1]
            if np.isinf(w.d):
                w.d = v.d + 1
                w.prev = v
                Q.append(w)

def DepthFirst(V, s=0, n=-1):
    """
    V: list of Nodes
    s: index of the initial node
    """
    V[s].c = "visited"
    S = [V[s]]

    while S:
        v = S.pop(n)
        print(v.state)
        for edge in v.adj:
            w = edge.nodes[1]
            if w.c == "unvisited":
                w.c = "visited"
                w.prev = v
                S.append(w)

def Dijkstra(V, s=0):
    RemoveUnreachable(V, s=s)
    V[s].d = 0
    S = [V[s]]

    while len(S) != len(V):
        # argmin{d(u) + l(u,v); u in S, v in S\V, (u,v) in E}を計算
        min, argmin = np.inf, None
        for u in S:
            for e in u.adj:
                if e.nodes[1] not in S:
                    if u.d + e.l < min:
                        argmin = e
                        min = u.d + e.l
        u, v = argmin.nodes
        v.d = min
        v.prev = u
        S.append(v)

def BellmanFord(V, s=0):
    # 負閉路なしと仮定
    V[s].opt[0] = 0
    for i in range(1, len(V)):
        for v in V:
            v.opt.append(min(v.opt[i-1], BellmanFord_min(v, V, i)))
    
def BellmanFord_min(v, V, i):
    min = np.inf
    for u in V:
        for e in u.adj:
            if e.nodes[1] == v:
                val = u.opt[i-1] + e.l
                if val < min:
                    min = val
    return min
    

def Path(s, t):
    """
    s: initial node
    t: terminal node
    """
    if s is t:
        return [s]
    return Path(s, t.prev) + [t]

    
def ConnectivityTest(V, how="b", s=0, display=False):
    Reset(V)
    if "depthfirst".startswith(how):
        n = -1
    elif "breadthfirst".startswith(how):
        n = 0
        
    V[s].d = 0
    Q = [V[s]]

    while Q:
        v = Q.pop(n)
        if display:
            print(v.state)
        for edge in v.adj:
            w = edge.nodes[1]
            if np.isinf(w.d):
                w.d = v.d + 1  
                Q.append(w)

def RemoveUnreachable(V, *args, **kwargs):
    ConnectivityTest(V, *args, **kwargs)
    for i in range(len(V)):
        if np.isinf(V[i].d):
            V.pop(i)
    Reset(V)
            
    

V1 = [Node(i) for i in range(1, 8)]
V1[0].connect([V1[1], V1[2], V1[3]])
V1[2].connect([V1[4]])
V1[3].connect([V1[5]])
V1[4].connect([V1[3]])
V1[5].connect([V1[4]])
V1[6].connect([V1[5]])

V2 = [Node(i) for i in range(13)]
V2[0].connect([V2[6], V2[1]])
V2[1].connect([V2[5], V2[2]])
V2[2].connect([V2[4], V2[3]])
V2[6].connect([V2[12], V2[8], V2[7]])
V2[8].connect([V2[11], V2[10], V2[9]])

V3 =  [Node(i) for i in range(5+1)]
V3[1].connect((V3[2], V3[4]),        (10, 5))
V3[2].connect((V3[3], V3[4]),        (1, 2))
V3[3].connect((V3[5],),              (4,))
V3[4].connect((V3[2], V3[3], V3[5]), (3, 9, 2))
V3[5].connect((V3[3],),              (6,))
V3.pop(0)
