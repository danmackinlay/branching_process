from sklearn import linear_model
import numpy as np

"""
Wrapping Lasso algorithms to provide weighting.
"""


class BaseWeightedRegression:

    RegressionModel = None

    def __init__(
            self,
            normalize=False,
            eps=1e-7,
            *args, **kwargs):
        """
        eps is used for rescaling divisions, and is distinct from,
        e.g. the epsilon in epsilon-insensitive Huber regression.
        """
        self.eps = eps
        self.unweighted = self.RegressionModel(
            normalize=normalize,
            *args, **kwargs
        )

    def _weight_X(self, X):
        return X / self.penalty_weight * self.sample_weight

    def _unweight_X(self, X):
        return X * self.penalty_weight / self.sample_weight

    def _weight_y(self, y):
        return y * self.sample_weight.ravel()

    def _unweight_y(self, y):
        return y / self.sample_weight.ravel()

    def _weight_coefs(self, coefs):
        return coefs * self.penalty_weight

    def _unweight_coefs(self, coefs):
        return coefs / self.penalty_weight

    @property
    def coef_(self):
        return self._unweight_coefs(self.unweighted.coef_)

    @property
    def intercept_(self):
        return self.unweighted.intercept_

    @property
    def alpha_(self):
        return self.unweighted.alpha_

    @property
    def n_iter_(self):
        return self.unweighted.n_iter_

    def _adaptive_weight(
            self,
            X,
            y,
            gamma=1.0,
            **weight_fit_arg):
        # choose adaptive Lasso weights
        regr = linear_model.Ridge(
            alpha=1.0,
            normalize=True,
        )
        regr.fit(X, y, **weight_fit_arg)
        abs_weight = 1.0/(np.abs(regr.coef_)**gamma + self.eps)
        return abs_weight

    def fit(
            self,
            X,
            y,
            gamma=1.0,
            sample_weight=None,
            penalty_weight=None,
            weight_fit_arg=None):
        if weight_fit_arg is None:
            weight_fit_arg = {}
        self.penalty_weight = np.ones((1, X.shape[1]))
        if penalty_weight == 'adaptive':
            self.penalty_weight[0, :] = self._adaptive_weight(
                X,
                y,
                gamma=gamma,
                **weight_fit_arg
            )
        elif penalty_weight == 'maladaptive':
            self.penalty_weight[0, :] = 1.0/self._adaptive_weight(
                X,
                y,
                gamma=gamma,
                **weight_fit_arg
            )
        elif penalty_weight is not None:
            self.penalty_weight[0, :] = penalty_weight.ravel()
        else:
            self.penalty_weight = np.ones(X.shape[1])
        self.sample_weight = np.ones((X.shape[0], 1))
        if sample_weight is not None:
            self.sample_weight[:, 0] = sample_weight.ravel()
        if sample_weight is not None:
            self.sample_weight[:, 0] = sample_weight.ravel()
        self.unweighted.fit(
            self._weight_X(X),
            self._weight_y(y)
        )
        return self

    def predict(self, X, penalty_weight=None, sample_weight=None):
        self.penalty_weight = penalty_weight or np.ones(X.shape[1])
        self.sample_weight = sample_weight or np.ones(X.shape[0])
        return self._unweight_y(
            self.unweighted.fit(
                self._weight_X(X)
            )
        )

    def score(self, X, y):
        return self.unweighted.score(X, y, self.sample_weight)


class WeightedLasso(BaseWeightedRegression):
    RegressionModel = linear_model.Lasso

    @property
    def sparse_coef_(self):
        return self.unweighted.sparse_coef_


class WeightedLassoLarsIC(BaseWeightedRegression):
    RegressionModel = linear_model.LassoLarsIC

    @property
    def criterion_(self):
        return self.unweighted.criterion_


class WeightedLassoCV(BaseWeightedRegression):
    RegressionModel = linear_model.LassoCV

    # LassoCV has a static method for calculating the whole path.
    # implementing it would require me to convert the coeffs
    # passed and returned.
    # def path(*args, **kwargs):

    @property
    def dual_gap_(self):
        return self.unweighted.dual_gap_

    @property
    def mse_path_(self):
        return self.unweighted.mse_path_

    @property
    def alphas_(self):
        return self.unweighted.alphas_


class WeightedLassoLarsCV(BaseWeightedRegression):
    RegressionModel = linear_model.LassoLarsCV

    @property
    def dual_gap_(self):
        return self.unweighted.dual_gap_

    @property
    def mse_path_(self):
        return self.unweighted.mse_path_

    @property
    def alphas_(self):
        return self.unweighted.alphas_

    @property
    def cv_alphas_(self):
        return self.unweighted.cv_alphas_

    @property
    def cv_mse_path_(self):
        return self.unweighted.cv_mse_path_

    @property
    def coef_path_(self):
        return self.unweighted.coef_path_
