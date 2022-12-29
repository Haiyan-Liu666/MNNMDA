import scipy.sparse
import os
import numpy as np
import matlab.engine
import time
import logging

from model_zoo.base import BaseModel


class MatlabModel(BaseModel):
    eng = None
    def __init__(self, matlab_file=None, workpath=None, **kwargs):
        super(MatlabModel, self).__init__(**kwargs)
        if not workpath:
            workpath = os.path.join(os.getcwd(), "model_zoo", "matlab_models")
        assert os.path.exists(workpath), f"{workpath} didn't exist!"
        # assert os.path.exists(os.path.join(workpath, matlab_file))
        self.workpath = workpath
        self.matlab_file = matlab_file

        self.eng = self.eng if self.eng is not None else matlab.engine.start_matlab()
        self.eng.addpath(self.workpath)


class MNNMDA(MatlabModel):
    def __init__(self, maxiter, alpha, beta, **kwargs):
        workpath = os.path.join(os.getcwd(), "model_zoo", "matlab_models", "MNNMDA")
        super(MNNMDA, self).__init__(workpath=workpath, **kwargs)
        self.maxiter = maxiter
        self.alpha = alpha
        self.beta = beta
        self.eng.workspace["maxiter"] = maxiter
        self.eng.workspace["alpha"] = alpha
        self.eng.workspace["beta"] = beta
        self.eng.workspace["tol1"] = 1e-1
        self.eng.workspace["tol2"] = 1e-2
        self.eng.workspace["a"] = 0.0
        self.eng.workspace["b"] = 1.0

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument("--maxiter", default=300, type=int)
        parent_parser.add_argument("--alpha", default=1.0, type=float)
        parent_parser.add_argument("--beta", default=100., type=float)
        return parent_parser

    def fit_transform(self, train_indices, test_indices, d_feature, m_feature, shape):
        interaction = scipy.sparse.coo_matrix((train_indices[:,-1], (train_indices[:,0], train_indices[:, 1])), shape=shape).todense()
        self.eng.workspace["Wdr"] = matlab.double(interaction.tolist())
        if m_feature.shape[0]!=m_feature.shape[1]:
            print("use GIPSim")
            dsim, msim = self.eng.eval("GIPSim(Wdr, 1, 1 )", nargout=2)
            self.eng.workspace["Wdd"] = dsim
            self.eng.workspace["Wrr"] = msim
        else:
            self.eng.workspace["Wdd"] = matlab.double(d_feature.tolist())
            self.eng.workspace["Wrr"] = matlab.double(m_feature.tolist())
        # print(np.trace(dsim),np.trace(msim))
        # self.eng.workspace["Wdd"] = matlab.double((np.array(dsim)/np.trace(dsim)+d_feature).tolist())
        # self.eng.workspace["Wrr"] = matlab.double((np.array(msim)/np.trace(msim)+m_feature).tolist())
        res = self.eng.eval("MNNMDA(Wrr, Wdr, Wdd, alpha, beta, tol1, tol2, maxiter, a, b)")
        res = np.array(res)
        return res



class KATZHMDA(MatlabModel):
    def __init__(self, beta, **kwargs):
        workpath = os.path.join(os.getcwd(), "model_zoo", "matlab_models", "KATZHMDA")
        super(KATZHMDA, self).__init__(workpath=workpath, **kwargs)
        self.beta = beta
        self.eng.workspace["beta"] = beta

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument("--beta", default=1.0, type=float)
        return parent_parser

    def fit_transform(self, train_indices, test_indices, d_feature, m_feature, shape):
        interaction = scipy.sparse.coo_matrix((train_indices[:,-1], (train_indices[:,0], train_indices[:, 1])), shape=shape).todense()
        self.eng.workspace["Wrr"] = matlab.double(m_feature.tolist())
        self.eng.workspace["Wdr"] = matlab.double(interaction.tolist())
        self.eng.workspace["Wdd"] = matlab.double(d_feature.tolist())
        res = self.eng.eval("KATZHMDA_py(Wrr, Wdr, Wdd, beta)")
        res = np.array(res)
        return res



class LRLSHMDA(MatlabModel):
    def __init__(self, lambdaM, lambdaD, lw, **kwargs):
        workpath = os.path.join(os.getcwd(), "model_zoo", "matlab_models", "LRLSHMDA")
        super(LRLSHMDA, self).__init__(workpath=workpath, **kwargs)
        self.lambdaM = lambdaM
        self.lw = lw
        self.eng.workspace["lambdaM"] = lambdaM
        self.eng.workspace["lambdaD"] = lambdaD
        self.eng.workspace["lw"] = lw

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument("--lambdaM", default=0.1, type=float)
        parent_parser.add_argument("--lambdaD", default=0.1, type=float)
        parent_parser.add_argument("--lw", default=0.1, type=float)
        return parent_parser

    def fit_transform(self, train_indices, test_indices, d_feature, m_feature, shape):
        interaction = scipy.sparse.coo_matrix((train_indices[:,-1], (train_indices[:,0], train_indices[:, 1])), shape=shape).todense()
        self.eng.workspace["Wrr"] = matlab.double(m_feature.tolist())
        self.eng.workspace["Wdr"] = matlab.double(interaction.tolist())
        self.eng.workspace["Wdd"] = matlab.double(d_feature.tolist())
        res = self.eng.eval("LRLSHMDA_py(Wrr, Wdr, Wdd, lambdaM, lambdaD, lw)")
        res = np.array(res)
        return res


class NTSHMDA(MatlabModel):
    def __init__(self, gamma, phi, delte, beta1, beta2, **kwargs):
        workpath = os.path.join(os.getcwd(), "model_zoo", "matlab_models", "NTSHMDA")
        super(NTSHMDA, self).__init__(workpath=workpath, **kwargs)
        self.gamma = gamma
        self.phi = phi
        self.delte = delte
        self.beta1 = beta1
        self.beta2 = beta2
        self.eng.workspace["gamma"] = gamma
        self.eng.workspace["phi"] = phi
        self.eng.workspace["delte"] = delte
        self.eng.workspace["beta1"] = beta1
        self.eng.workspace["beta2"] = beta2

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument("--gamma", default=0.7, type=float)
        parent_parser.add_argument("--phi", default=0.9, type=float)
        parent_parser.add_argument("--delte", default=0.3, type=float)
        parent_parser.add_argument("--beta1", default=0.8, type=float)
        parent_parser.add_argument("--beta2", default=0.2, type=float)
        return parent_parser

    def fit_transform(self, train_indices, test_indices, d_feature, m_feature, shape):
        interaction = scipy.sparse.coo_matrix((train_indices[:,-1], (train_indices[:,0], train_indices[:, 1])), shape=shape).todense()
        self.eng.workspace["interaction"] = matlab.double(interaction.tolist())
        res = self.eng.eval("NTSHMDA_py(gamma, phi, delte, beta1, beta2, interaction)")
        res = np.array(res)
        return res


