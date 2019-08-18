import math
import numpy as np

# This gets the implementation of vec2angle, angle2vec and move which
# moves solid angles correct.

def dot(v, w):
    assert len(v) == len(w)
    return sum([v[i] * w[i] for i in range(len(v))])

def vlensq(v):
    """squared length"""
    return dot(v, v)

def vlen(v):
    """length"""
    return math.sqrt(vlensq(v))

def normalize(v):
    """normalized vector"""
    l = len(v)
    return [x / l for x in v]

def vec2angle(v):
    """convert a vector to angles. Returns tan(theta)"""
    # x0 = cos t0
    # x1 = sin t0 cos t1
    # x2 = sin t0 sin t1 cos t2
    # ...
    # x{n-4} = sin t0 sin t1 ... sin t{n-5} cos t{n-4}
    # x{n-3} = sin t0 sin t1 ... sin t{n-5} sin t{n-4} cos t{n-3}
    # x{n-2} = sin t0 sin t1 ... sin t{n-5} sin t{n-4} sin t{n-3} cos t{n-2}
    # x{n-1} = sin t0 sin t1 ... sin t{n-5} sin t{n-4} sin t{n-3} sin t{n-2}
    # contains tan(t(n-2)) at final index, and tan(tn) . cos(n+1) for all other indeces
    n = len(v)
    a = [0 for _ in range(len(v) - 1)]
    sinaccum = 1
    for i in range(len(v) - 1):
        if sinaccum != 0:
            a[i] = math.acos(v[i] / sinaccum)
        else:
            a[i] = 0
        sinaccum *= math.sin(a[i])
    return a

def angleBwVec(v, w):
    n = len(v)
    a = [0 for _ in range(len(v) - 1)]


def angle2vec(a):
    v = [0 for _ in range(len(a) + 1)]
    sinaccum = 1
    for i in range(len(a)):
        v[i] = sinaccum * math.cos(a[i])
        sinaccum *= math.sin(a[i])
    v[len(a)] = sinaccum
    checkunit(v)
    return v

def checkunit(v):
    assert abs(vlen(v) - 1) < 1e-2 


def projectplane(v, i):
    """project a vector v onto the plane with normal
        (0, ..., 1 (at i), ...0, )
    """
    return [v[j] if j != i else 0 for j in range(len(v))]

def clamp(v):
    return [0 if abs(v[i]) < 1e-5 else v[i] for i in range(len(v))]

def move(v1, v2, v3):
    """v1 : v2 :: v3 : output"""
    a1 = vec2angle(v1)
    a2 = vec2angle(v2)
    a3 = vec2angle(v3)
    return angle2vec([a2[i] - a1[i] + a3[i] for i in range(len(a1))])

def closest(v, vs):
    bestdot = 0
    for i in range(1, len(vs)):
        dot = math.abs(1.0 - dot(v, vs[0]))
        if dot > bestdot: bestix = i

    return vs[i]

if __name__ == "__main__":
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    vecs = { "x": x, "y": y, "z": z }
    def debug(name, v):
        print("%s: %s | angle : %s | vec from angle: %s " % (name, v, vec2angle(v), angle2vec(vec2angle(v))))
    for (name, v) in vecs.iteritems():
        debug(name, v)

    for (n1, v1) in vecs.iteritems():
        for(n2, v2) in vecs.iteritems():
            for(n3, v3) in vecs.iteritems():
                print("%s (%s) : %s (%s) :: %s (%s) : %s" % (n1, v1, n2, v2, n3, v3, clamp(move(v1, v2, v3))))
