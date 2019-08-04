#!/usr/bin/env python
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
        e = Expr()
        e.type = "cos"
        e.inner = inner
        return e

    @classmethod
    def sin(cls, inner):
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
                    return Expr(l.v * r.v)
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



e =  Expr.cos(Expr("x")) * Expr.cos(Expr("y"))
print(e)
print("de/dy: %s " % e.der("y").simpl())
