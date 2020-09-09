import numpy as np


class Kernel:
    def __init__(self, noise=0):
        self.params = dict()
        self.params["noise"] = noise

    def __call__(self, x1, x2):
        return self.K(x1, x2)

    def K(self, x1, x2):
        pass

    def dK_dl(self, X):
        pass

    def dK_dsigma(self, X):
        return 2 * self.K(X, X) / self.params["sigma"]

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.params.keys():
                self.params["{}".format(k)] = v

    def condition_f_observation(self, X, Y):
        N = np.max(X.shape)
        G = self.K(X, X) + self.params["noise"] ** 2 * np.eye(N)
        self.L = np.linalg.cholesky(G)
        self.alpha = self.K_inv_mult(Y)

        self.log_ml = -0.5 * np.sum(Y * self.alpha) - np.sum(np.log(np.diag(self.L)))
        return self.alpha, self.log_ml

    def K_inv_mult(self, Y):
        return np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))

    def __repr__(self):
        specification = self.__class__.__name__ + "["
        for k, v in self.params.items():
            specification += "{}:{} ".format(k, v)
        return specification + "]"


def test_dK_dl(kern, eps=1e-3, verbose=False):
    """Utility function to test dK_dl with central difference approximation of gradient"""
    ell = kern.params["ell"]
    X = np.arange(4).reshape(-1, 1) * ell

    # Implemented function
    dk_dl = kern.dK_dl(X)

    # Central difference approximation
    kern.update_params(ell=ell - eps)
    K1 = kern.K(X, X)
    kern.update_params(ell=ell + eps)
    K2 = kern.K(X, X)
    kern.update_params(ell=ell)
    central_diff = (K2 - K1) / (2 * eps)

    print(kern)
    print("\n==================\n")
    if verbose:
        print("Implemented derivative:\n", dk_dl)
        print("Numerical differentiation:\n", central_diff)

    avg_dist = np.linalg.norm(central_diff - dk_dl) / (len(X) ** 2)
    if avg_dist < 10 * eps ** 2:
        print("Derivative seems correct\n")
    else:
        print("Derivative not within error tolerance\n")

    print(
        "average deviation from numeric gradient < 10 * eps^2: [{}]\n  average deviation:[{:.1e}] \n          tolerance:[{:.1e}]\n".format(
            avg_dist < 10 * eps ** 2, avg_dist, 10 * eps ** 2
        )
    )
