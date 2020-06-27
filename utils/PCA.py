import numpy as np 
def PCA(X, K):
    cov_matrix = np.cov(X.T)    
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
    
    sortIdx = eigenValues.argsort()[::-1]

    eigenValues_Topk = eigenValues[sortIdx][:K]
    eigenVectors_Topk = eigenVectors[:, sortIdx][:, :K]

    X_ = X.dot(eigenVectors_Topk)
    return X_


