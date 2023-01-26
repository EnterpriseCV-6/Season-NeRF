import numpy as np
from scipy.integrate import quad

class _spline_part():
    def __init__(self, X0, X1, A):
        self.X0 = X0
        self.X1 = X1
        self.A = A
        self.n = A.shape[0]
        self.first = False
        self.last = False

    def __call__(self, X):
        Y = 0
        h = X - self.X0
        for i in range(self.n):
            Y += self.A[i] * (h)**i
        return Y

    def full_eval_with_diff(self, X):
        Y = np.zeros(self.n)
        h = X - self.X0

        Cs = np.ones(self.n, dtype=int)
        Ps = np.arange(0, self.n)
        for j in range(self.n):
            for i in range(self.n-j):
                Y[j] += self.A[i+j] * (h) ** i * Cs[i+j]
            Cs = Cs * Ps
            Ps = Ps - 1
        # exit()
        # for i in range(self.n):
        #     # Y[0] += self.A[i] * (h) ** i
        #     for j in range(self.n-i-1, -1, -1):
        #         print(j)
        #         Y[j] += self.A[j] * h ** i * self._fact(j)
        # print(self.A)
        # print(Y)
        # exit()
        return Y

    def safe_call(self, X):
        if (self.X0 <= X < self.X1) or (self.last and self.X0 <= X) or (self.first and X < self.X1):
            return self(X)
        else:
            return 0

    def full_safe_call(self, X):
        if (self.X0 <= X < self.X1) or (self.last and self.X0 <= X) or (self.first and X < self.X1):
            return self.full_eval_with_diff(X)
        else:
            return np.zeros(self.n)

    def _fact(self, j:int):
        if j == 0:
            return 1
        elif j == 1:
            return 1
        else:
            return j * self._fact(j-1)

    # def get_arc_length(self, Start, End):
    #     X0 = min(Start, End)
    #     X1 = max(Start, End)
    #     if (self.X0 > X0 and self.X0 >= X1) or (self.X1 < X0 and self.X1 < X1):
    #         ans = (0, -1)
    #     else:
    #         sign = 1 if Start < End else -1
    #         X0 = max(X0, self.X0)
    #         X1 = min(X1, self.X1)
    #
    #         ans = sign * quad(self, X0, X1)
    #     return ans


class _spline():
    def __init__(self, X, Y):
        self.parts_list = []
        self.X = X
        self.Y = Y

    def build(self):
        pass

    def __call__(self, X):
        return np.sqrt(1+self.slow_eval(X)**2)

    def slow_eval(self, X):
        ans = 0
        for a_part in self.parts_list:
            ans += a_part.safe_call(X)
        return ans

    def full_slow_eval(self, X):
        ans = 0
        for a_part in self.parts_list:
            ans += a_part.full_safe_call(X)
        return ans

    # def get_length(self, X0, X1):
    #     ans = 0
    #     for a_part in self.parts_list:
    #         ans += a_part.get_arc_length(X0, X1)
    #     return ans

    def get_arc_length(self, Start, End):
        ans = quad(self, Start, End)
        return ans

class spline_3(_spline):
    def __init__(self, X, Y, start_end_dx = (None, 0, 0)):
        super(spline_3, self).__init__(X, Y)
        # if start_end_dx[0] is None:
        #     self.start_dx = (Y[1]-Y[0])/(X[1] - X[0])
        # else:
        #     self.start_dx = start_end_dx[0]
        # if start_end_dx[1] is None:
        #     self.start_dx_dx = ((Y[2]-Y[0])/(X[2]-X[0]) - self.start_dx)/(X[2] - X[0])
        # else:
        #     self.start_dx_dx = start_end_dx[1]
        # if start_end_dx[2] is None:
        #     self.start_dx_dx_dx = 0
        # else:
        #     self.start_dx_dx_dx = start_end_dx[2]

        self.build()


    def build(self):
        A = np.zeros([self.X.shape[0], self.X.shape[0]])
        Y = np.zeros([self.X.shape[0]])
        for i in range(1,self.X.shape[0]-1):
            hi, hi1 = self.X[i] - self.X[i-1], self.X[i+1] - self.X[i]
            A[i,i-1], A[i, i], A[i, i+1] = hi, 2*(hi+hi1), hi1
            Y[i] = 3*((self.Y[i+1] - self.Y[i]) / hi1 - (self.Y[i] - self.Y[i-1]) / hi)

        # #natural_condition
        Y[0], Y[-1] = 0, 0
        A[0,0], A[0,1] = 1, -1
        A[-1, -2], A[-1, -1] = 1, -1

        # A = A[1:-1, 1:-1]
        # Y = Y[1:-1]
        # X = np.linalg.inv(A) @ Y
        # X = np.concatenate([np.array([0]), X, np.array([0])])

        # #Circular Condition
        # hi, hi1 = self.X[-1] - self.X[-2], self.X[1] - self.X[0]
        # A[0,  -1], A[0, 0], A[0, 1] = hi, 2 * (hi + hi1), hi1
        # Y[0] = 3 * ((self.Y[1] - self.Y[0]) / hi1 - (self.Y[-1] - self.Y[-2]) / hi)
        #
        # hi, hi1 = self.X[-1] - self.X[-2], self.X[1] - self.X[0]
        # A[-1, -2], A[-1, -1], A[-1, 0] = hi, 2 * (hi + hi1), hi1
        # Y[-1] = 3 * ((self.Y[1] - self.Y[0]) / hi - (self.Y[-1] - self.Y[-2]) / hi1)
        # X = np.linalg.inv(A) @ Y


        # # not a knot
        # A[0, 0], A[0, 1] = 1, -1
        # A[-1, -2], A[-1, -1] = 1, -1
        # Y[0] = 0
        # Y[-1] = 0
        # X = np.linalg.inv(A) @ Y



        # A = A[0:-1, 0:-1]
        # Y = Y[0:-1]
        X = np.linalg.inv(A) @ Y
        # X = np.concatenate([X, np.array([0])])

        for i in range(self.X.shape[0]-1):
            hi = self.X[i+1] - self.X[i]
            a = self.Y[i]
            b = (self.Y[i+1] - self.Y[i]) / hi - hi/3 * (2*X[i] + X[i+1])
            c = X[i]
            d = (X[i+1] - X[i]) / (3*hi)
            A = np.array([a,b,c,d])
            # print(A)
            self.parts_list.append(_spline_part(self.X[i], self.X[i+1], A))
        self.parts_list[0].first = True
        self.parts_list[-1].last = True