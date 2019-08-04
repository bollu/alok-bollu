#!/usr/bin/env python3
import sympy
from itertools import *
from sympy import *
import sympy as sym
import sh
from random import randrange

class Expr:
    def __init__(self, v=None):
        if v is None:
            self.type = "undef"
            return
        if type(v) is float or type(v) is int:
            self.v = v
            self.type = "num"
            return
        elif type(v) is str:
            self.v = v
            self.type = "id"
            return
        
        raise Exception("unknown expression: ", v)
    def __eq__(self, other):
        if self.type == "num" or self.type == "id":
            return self.type == other.type and self.v == other.v
        elif self.type in ["add", "sub", "mul", "div"]:
            return self.type == other.type and self.l == other.l and self.r == other.r
        elif self.type in ["sin", "cos"]:
            return self.type == other.type and self.inner == other.inner


    def __add__(self, o):
        e = Expr()
        e.type = "add"
        e.l = self
        e.r = o
        return e

    def __sub__(self, o):
        e = Expr()
        e.type = "sub"
        e.l = self
        e.r = o
        return e

    def __mul__(self, o):
        e = Expr()
        e.type = "mul"
        e.l = self
        e.r = o
        return e

    @classmethod
    def cos(cls, inner):
        assert(isinstance(inner, Expr))
        e = Expr()
        e.type = "cos"
        e.inner = inner
        return e

    @classmethod
    def sin(cls, inner):
        assert(isinstance(inner, Expr))
        e = Expr()
        e.type = "sin"
        e.inner = inner
        return e

    def __str__(self):
        if self.type == "num" or self.type == "id":
            return str(self.v)
        elif self.type == "add":
            return "(+ %s %s)" % (self.l, self.r)
        elif self.type == "sub":
            return "(- %s %s)" % (self.l, self.r)
        elif self.type == "mul":
            return "(* %s %s)" % (self.l, self.r)
        elif self.type == "div":
            return "(/ %s %s)" % (self.l, self.r)
        elif self.type == "sin" or self.type == "cos":
            return "(%s %s)" % (self.type, self.inner)
    def __repr__(self):
        return self.__str__()

    def tosympy(self):
        if self.type == "id":
            return sym.Symbol(self.v)
        if self.type == "num":
            return self.v
        if self.type == "add":
            return self.l.tosympy() + self.r.tosympy()
        if self.type == "mul":
            return self.l.tosympy() + self.r.tosympy()
        if self.type == "sin":
            return sym.sin(self.inner.tosympy())
        if self.type == "cos":
            return sym.cos(self.inner.tosympy())
        raise Exception("unknown tosympy")

    def der(self, v):
        if self.type == "num":
            return Expr(0)
        elif self.type == "id":
            return Expr(1) if self.v == v else Expr(0)
        elif self.type == "add":
            return self.l.der(v) + self.r.der(v)
        elif self.type == "sub":
            return self.l.der(v) - self.r.der(v)
        elif self.type == "mul":
            return self.l * self.r.der(v) + self.r * self.l.der(v)
        elif self.type == "sin":
            return Expr.cos(self.inner) * self.inner.der(v)
        elif self.type == "cos":
            return Expr(-1) * Expr.sin(self.inner) * self.inner.der(v)

    def simpl(self):
        if self.type in ["add", "sub", "mul"]:
            l = self.l.simpl()
            r = self.r.simpl()
            if self.type == "add":
                if l == Expr(0):
                    return r
                elif r == Expr(0):
                    return l
                elif l.type == "num" and r.type == "num":
                    return Expr(l.v + r.v)
                else:
                    return l + r
            elif self.type == "mul":
                if l == Expr(1):
                    return r
                elif r == Expr(1):
                    return l
                elif l == Expr(0) or r == Expr(0):
                    return Expr(0)
                elif l.type == "num" and r.type == "num":
                    return Expr(l.v * r.v)
                else:
                    return l * r
        if self.type in ["sin", "cos"]:
            inner = self.inner.simpl()
            if self.type == "sin":
                return Expr.sin(inner)
            elif self.type == "cos":
                return Expr.cos(inner)
        return self
# convert angles to vectors
def angle2vec(angle):
    sinaccum = Expr(1)
    vec = []
    for a in angle:
        vec.append((sinaccum * Expr.cos(a)))
        sinaccum *= Expr.sin(a)
    vec.append(sinaccum)
    return vec

def vec2angleder(vec):
    angleders = [Expr(0) for _ in range(len(vec))]
    for i in range(len(vec)):
        angleders[i] += vec[i].der("a" + str(i)).simpl().simpl()


    return list(map(lambda e: e.simpl(), angleders))

if __name__ == "__main__":
    n = 4
    angles = [sym.Symbol("a"+str(i)) for i in range(n-1)]
    anglevec = []
    sinaccum = 1
    for a in angles:
        anglevec.append(sinaccum * sym.cos(a))
        sinaccum *= sym.sin(a)
    anglevec.append(sinaccum)
    print("anglevec: ", anglevec)

    lensq = 0
    for v in anglevec:
        lensq += v * v
    lensq = simplify(lensq)
    assert abs(lensq.evalf() - 1) == 0

    vec = [(i+1) for i in range(n)]
    print("vec: ", vec)

    dot = 0
    for i in range(len(vec)):
        dot += vec[i] * anglevec[i]
    print("dot: ", dot)

    # total derivative of the dot product wrt to the angle, in order of angle
    ders = [diff(dot, a) for a in angles]
    print("derivatives: ", ders)

    print("----")
    NTEST = 1000
    word2vec = sh.Command("./word2vec")
    for _ in range(NTEST):
        # angle substitutions
        anglevals = [randrange(-10, 10) for _ in range(n-1)]
        # subs to be fed to sympy
        subs = [(angles[i], anglevals[i]) for i in range(n-1)]
        
        # derivative values
        dervals = [float(simplify(der.subs(subs))) for der in ders]

        inp = [n] + vec + anglevals
        print("input: ", inp) 

        outvals = word2vec("-stress-test", *inp)

        print("---raw output---")
        print(outvals)
        print("---(done)---")

        # only one line should have angles_der: f1 f2 ... fn
        out_angles_der = [l for l in outvals.split("\n") if l.find("angles_der:") != -1][0]
        outvals = list(map(float, out_angles_der.split("angles_der:")[1].split()))

        print("*  dervals(reference): ", dervals)
        print("*  outvals(from word2vec):", outvals)

        for (der, out) in zip(dervals, outvals):
            delta = abs(der - out)
            print("*  |%s - %s| = %s" % (der, out, delta))
            assert delta < 1e-2
