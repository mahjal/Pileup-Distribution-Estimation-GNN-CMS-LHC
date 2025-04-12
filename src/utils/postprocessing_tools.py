import numpy as np
import scipy

def llikelihood_pois(lam, x):
#     print(f'lam = {lam}') # there is some nan in lam
#     print(f'x = {x}')
    return (x * np.log(lam) - lam - x*np.log(x) + x).sum()

def max_log_likelihood(x, loglikelihood_func, init_guess):
    
    def helper(p, p0):
#         print(f"p: {p}, p0 : {p0})")
        L0 = loglikelihood_func(p0, x)
#         print(f'helper = {loglikelihood_func(p, x) - 1}' )
        return L0 - loglikelihood_func(p, x) - 1  # this return some nan and zero value
    
    
    mle = x.mean()
    res_r = scipy.optimize.root(helper, mle + 0.1, args=(mle,))
    res_l = scipy.optimize.root(helper, mle - 0.1, args=(mle,))
    
    success = res_r.success and res_l.success
#     print(f"MLE: {mle}, 95% CI: ({res_l}, {res_r})")
    
    if not success:
        raise RuntimeError("Error interval Computation Failed")
    
    return mle - res_l.x, mle, res_r.x - mle


def R_squared(y, yhat, y_prime=None):
    RSS = (y - yhat)**2
    
    if y_prime is not None:
        RSS_prime = (yhat - y_prime) ** 2
    else:
        RSS_prime = (yhat - y.mean()) ** 2
    
    return 1 - RSS/RSS_prime


def remove_nan_node(node_features, edge_index, edge_attributes):    
    node_ID, feature_ID = np.where(np.isnan(node_features))
    node_features       = np.delete(node_features, node_ID, axis=0)
    
    for node in node_ID:
        _, edge    = np.where(edge_index == node)
        edge_index = np.delete(edge_index, edge, axis=1)
        edge_attributes = np.delete(edge_attributes, edge, axis=0)
    
    return node_features, edge_index, edge_attributes


def remove_inf_node(node_features, edge_index, edge_attributes):    
    node_ID, feature_ID = np.where(np.isinf(node_features))
    node_features       = np.delete(node_features, node_ID, axis=0)
    
    for node in node_ID:
        _, edge    = np.where(edge_index == node)
        edge_index = np.delete(edge_index, edge, axis=1)
        edge_attributes = np.delete(edge_attributes, edge, axis=0)
    
    return node_features, edge_index, edge_attributes
