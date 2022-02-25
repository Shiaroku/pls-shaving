import numpy as np
from operator import itemgetter
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


def gene_shaving(X, y, indices : list = None, alpha : float = 0.05, size : int = 5, pls = PLSRegression(n_components=1)) -> list:
    """ Takes an X matrix (size p x n) and a y vector of size p and 
    returns indices of the top k discriminant variables amongst the n candidates """
    p, n = X.shape
    assert y.shape[0] == p, "X must be of shape (p, n) and y of matching shape (p,)"

    if not indices:
        indices = np.arange(n)

    if n <= size:
        return indices
    else:
        pls.fit(X,y)
        rev_order = np.argsort([x[0]**2 for x in pls.x_weights_])[::-1]
        shaved_rev_order = rev_order[:int(max(np.floor((1-alpha)*n), size))]
        return gene_shaving(X=pd.DataFrame(X).iloc[:, shaved_rev_order], 
                            y=y, indices=list(itemgetter(*shaved_rev_order)(indices)), 
                            alpha=alpha, size=size, pls=pls)



### Dummy Example ###

X = np.random.rand(10, 1000)
y = np.random.binomial(1, 1/2, 10)
top_5_indices = gene_shaving(X, y)

print(top_5_indices)
