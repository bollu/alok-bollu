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

print(Expr(1) + Expr(2) * Expr("x") * Expr.cos(Expr("x")))
