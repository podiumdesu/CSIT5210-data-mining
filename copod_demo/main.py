import numpy as np
import pandas as pd

class COPOD():
    def __init__(self, threshold):
        super(COPOD, self).__init__(threshold=threshold)
        
    # Call ecdf func
    def ecdf(self, X):
        ecdf = ECDF(X)
        return ecdf(X)
        
    # Compute empirical copula observations
    def get_copula_observations(self, X, type):
        # Compute left tail ECDFs
        if type == "left":
            self.U = np.apply_along_axis(self.ecdf, 0, X)
            return self.U
        # Compute right tail ECDFs
        elif type == "right":
            self.V = np.apply_along_axis(self.ecdf, 0, -X)
            return self.V
        # Compute Skewness Corrected ECDFs
        elif type == "skewness":
            # Compute bi for Skewness Correction
            self.b = np.sign(np.apply_along_axis(skew, 0, X))
            # if b<0 W = U | if b>0 W = V
            self.W = self.U * -1 * np.sigh(self.b-1) + self.V * np.sign(self.b+1)
            return self.W

    def outlier_out(self, X):
        # Compute the tail probabilities
        self.left = pd.DataFrame(-1*np.log(self.get_copula_observations(X, "left")))
        self.right = pd.DataFrame(-1*np.log(self.get_copula_observations(X, "right")))
        self.skewness = pd.DataFrame(-1*np.log(self.get_copula_observations(X, "skewness")))

        # choose the greatest tail probability
        self.max_out = np.maximum(1/2*(self.left+self.right), self.skewness)

        self.ol_scores = self.max_out.sum(axis=1).to_numpy()

        # tag the label
        # 1 means outlier, 0 means non-outlier
        for i in range(len(self.ol_scores)):
            if self.ol_scores[i] >= self.threshold:
                self.labels[i] = 1
            else:
                self.labels[i] = 0 
        return self.labels