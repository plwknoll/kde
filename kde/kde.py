import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

class KDE:
    
    def __init__(self, obs, h = None, cutoff = None):
        self.obs = np.array(obs, dtype = "float64")
        self.n = len(obs)
        self.cutoff = cutoff
        
        self.inv_cdf = np.vectorize(self.inv_cdf)
        
        if h is not None:
            self.h = h
        else:
            iqr = np.subtract(*np.percentile(self.obs, [75, 25]))
            iqr = np.inf if np.isclose(iqr,0) else iqr
            std = np.std(self.obs)
            A = min(std, iqr/1.34)
            self.h = 1.059 * A * self.n ** (-1/5.)
        
        if self.cutoff is not None:
            self.left = self.cutoff[0]
            self.right = self.cutoff[1]
            
            self.leftval = self.uncut_cdf(self.left)
            self.rightval = self.uncut_cdf(self.right)
        
    def kernel(self, x):
        return (1/np.sqrt(2*np.pi)) * np.exp(-0.5*np.power(x,2))
    
    def pdf(self,x):
        if self.cutoff is None:
            return self.uncut_pdf(x)
        
        else:
            res = self.uncut_pdf(x) / (self.rightval - self.leftval)
            cut_mask = np.logical_or((x < self.left), (x > self.right))
            res[cut_mask] = 0
            return res
    
    def uncut_pdf(self,x):
        x = np.array(x, dtype = "float64")
        return self.kernel((x.reshape(-1,1) - self.obs) / self.h).mean(axis=1) / self.h
    
    def cdf(self,x):
        if self.cutoff is None:
            return self.uncut_cdf(x)
        else:
            res = (self.uncut_cdf(x) - self.leftval) / (self.rightval - self.leftval)
            res[res < 0] = 0
            res[res >= 1] = 1

            return res
    
    def uncut_cdf(self, x):
        x = np.array(x, dtype = "float64")
        
        return ndtr((x.reshape(-1,1) - self.obs) / self.h).mean(axis=1)
    
    def inv_cdf(self,x):
        min_obs = min(self.obs)
        max_obs = max(self.obs)
        mid_obs = (min_obs + max_obs) / 2
        return minimize((lambda s, t : (self.cdf(s) - t)**2), x0=mid_obs, args = x, bounds=[(min_obs, max_obs)], tol=1E-10).x[0]