# Graph representation
V = [1, 2, 3, 4, 5]
E = [({V[0], V[1]}, 1), # [(edge, weight), ...]
     ({V[0], V[4]}, 2),
     ({V[1], V[2]}, 6),
     ({V[1], V[3]}, 4),
     ({V[1], V[4]}, 2),
     ({V[2], V[3]}, 5),
     ({V[3], V[4]}, 3)] 

def Kruskal(V, E):
    """
    Kruskal's algorithm for a minimum spanning tree.
    V: a set of vertices
    E: a set of edges
    """

    E.sort(key=lambda e: e[-1])

    T = []
    c = {v : i for v, i in zip(V, range(len(V)))}

    for e in E:
        u, v = e[0]
        if c[u] != c[v]:
            T.append(e)
            for node in V:
                if c[node] == c[v]:
                    c[node] = c[u]
        
    return T
