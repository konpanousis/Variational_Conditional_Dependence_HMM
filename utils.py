"""
Various util functions for the implementation.

@author: Konstantinos P. Panousis
Cyprus University of Technology
"""

import torch
import numpy as np
import scipy
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

from scipy.sparse import csr_matrix
import sys

# set seed
torch.manual_seed(0)
np.random.seed(0)


def converge(log_probs, cur_log_prob, verbose = False):
    """
    Check convergence of the EM like approach of CD-HMM.

    :param log_probs: the list of log probabilities for each iter
    :param cur_log_prob: the cur log probability given the current estimates of the parameters
    :param verbose: flag to print stuff, boolean
    :return: converged: boolean, save flag: boolean
    """
    converged = False
    save = False

    if len(log_probs) > 1:

        diff = np.abs(cur_log_prob.numpy()) - np.abs(log_probs[-1])

        if diff >= 0 > cur_log_prob.numpy():
            if diff < 300:
                if verbose:
                    print('---------------------------------------')
                    print('Slightly Decreasing bound. Diff:', diff)
                    print('---------------------------------------')
                log_probs.append(cur_log_prob.numpy())
            elif diff >= 300.1:
                if verbose:
                    print('\nDECREASING BOUND\n')
                    print('Current:', cur_log_prob.numpy())
                    print('Previous:', log_probs[-1])
                    print(cur_log_prob.numpy() - log_probs[-1])
                log_probs.append(cur_log_prob.numpy())

        if np.abs(log_probs[-2] - log_probs[-1]) < 1e-10:
            print('Converged at iter', iter, 'with LL:', log_probs[-1])
            log_probs.append(cur_log_prob.numpy())
            converged = True

        elif cur_log_prob.numpy() >= np.max(log_probs):
            log_probs.append(cur_log_prob.numpy())
            save = True

    else:
        log_probs.append(cur_log_prob.numpy())

    return converged, save


###########################################
##### INITIALIZATION
##############################################
def init(x, K, M, D):
    """
    Function for initializing the posterior parameters through KMeans

    :param x: the data, np array
    :param K: the number of states
    :param M: the number of components
    :param D: the dimensionality of the data
    :return: the inital values of the parameters
    """

    # number of sequences
    N = x.shape[0]

    # get time frames and run kmeans
    first_slice_idx = np.zeros([N], dtype=np.int32)
    crt_first_slice_idx = 0
    all_samples = np.zeros([D, 1000*N])

    for i in range(N):
        first_slice_idx[i] = crt_first_slice_idx
        crt_first_slice_idx = crt_first_slice_idx + x[i].shape[1]
        all_samples[:, first_slice_idx[i]:crt_first_slice_idx] = x[i]

    all_samples = all_samples[:, :crt_first_slice_idx]

    tries = 0
    print('Kmeans Try: 1')
    idx = KMeans(n_clusters=K).fit(all_samples.T).labels_

    while np.bincount(idx).min() < M:
        tries += 1
        idx = KMeans(n_clusters=K, init='random').fit(all_samples.T).labels_

        # if KMeans fails, try with random init
        if tries == 2000:
            print('tried init 2000 times to get idx and failed..')
            return 0.,0,0,0,0,0,0, False

    prior_all = np.zeros([N, K])
    transmat_all = np.zeros([N, K, K])
    transmat_all_count = np.zeros([K, K])

    # the mu, sigma and weights are initialized either with cov of data or with GMMs
    mu = np.zeros([K, M, D])
    sigma = np.zeros([K, M, D, D])
    weights = np.zeros([K, M])

    for i in range(K):
        mu[i, :, :], sigma[i, :, :, :], weights[i, :] = mixgauss_init(all_samples[:, idx == i], M)

    for i in range(N):
        if i < N-1:
            IDX_i = idx[first_slice_idx[i]:first_slice_idx[i+1]]
        else:
            IDX_i = idx[first_slice_idx[i]:crt_first_slice_idx]

        prior_all[i, IDX_i[0]] = 1
        transmat_all[i, :, :], c = est_trans(IDX_i, K)
        transmat_all_count += c

    #  init distro second layer process
    prior = prior_all.sum(0, keepdims = True)/N
    transmat = transmat_all_count/transmat_all_count.sum(-1, keepdims = True)

    return prior, transmat,\
           prior_all, transmat_all, mu, sigma, weights, True


def est_trans(idx, K):
    """
    Estimate the transition probabilities using the counts in clusters
    :param idx: the indices
    :param K: the number of states
    :return: probabilites, counts
    """

    counts = np.zeros([K, K])
    for i in range(len(idx)-1):
        counts[idx[i], idx[i+1]] +=1

    counts_norm = counts.sum(-1, keepdims = True)
    counts_norm = np.where(counts_norm == 0, np.ones_like(counts_norm), counts_norm)

    return counts/counts_norm, counts

def mixgauss_init(data, M, method = 'rand', cov_type = 'full'):
    """
    Used to initialize the emission distribution
    :param data: the data
    :param M: the number of mixtures
    :param method: random or GMM
    :param cov_type: choose covariance type
    :return: mu, sigma, weights
    """

    d, T = data.shape

    if method == 'rand':
        s = np.cov(data)
        #s += 0.01*np.random.uniform(s.shape[0])
        sigma = np.tile((0.5*np.diag(np.diag(s)))[None, :, :], [M, 1, 1])

        indices = np.random.permutation(T)

        mu = data[:, indices[:M]]
        weights = np.ones([1, M])/M

    elif method == 'kmeans':
        gmm = GMM(n_components=M).fit(data.T)
        mu = gmm.means_.T
        sigma = gmm.covariances_
        weights = gmm.weights_

    if cov_type == 'diag':
        for i in range(M):
            sigma[i, :, :] = np.diag(np.diag(sigma[i]))

    return mu.T, sigma, weights


##################
### Distro stuff
##################

def expectation_log_dirichlet(pi):
    """
    The expectation of the log of a dirichlet rv
    :param pi:  the current values of the parameters
    :return:  the expected log value
    """
    return torch.digamma(pi) - torch.digamma(torch.sum(pi, -1, keepdim=True))


def dirichlet_kl(prior_a, posterior_a):
    """
    The KL divergence for the Dirichlet distribution
    :param prior_a: the prior parameters of the distribution
    :param posterior_a: the parameters of the posterior distribution
    :return: the KL divergence
    """

    sum_prior_a = prior_a.sum(-1, keepdim=True)
    sum_posterior_a = posterior_a.sum(-1, keepdim=True)
    t1 = sum_posterior_a.lgamma() - sum_prior_a.lgamma()
    t2 = (prior_a.lgamma() - posterior_a.lgamma()).sum(-1, keepdim=True)
    t3 = posterior_a - prior_a
    t4 = posterior_a.digamma() - sum_posterior_a.digamma()

    return t1 + t2 + (t3 * t4).sum(-1, keepdim = True)


def niw_kl(t_m, m, t_l, l, t_eta, eta, t_s, s):
    """
    The KL distribution for the imposed NW distribution.
    :param t_m: the posterior m parameter
    :param m: the prior m parameter
    :param t_l: the posterior lambda parameter
    :param l: the prior lambda parameter
    :param t_eta: the posterior eta parameter
    :param eta: the prior eta parameter
    :param t_s: the posterior S parameter
    :param s: the prior S parameter
    :return: The KL divergence
    """

    N, K, D, _ = s.shape
    temp = t_m - m
    temp4 = torch.zeros([N, K])

    for i in range(N):
        for k in range(K):
            chol = torch.cholesky(t_eta[i, k]*t_s[i, k])
            M = batch_mahalanobis(chol, temp[i,k])

            temp4[i, k] =   M
            sz0, sz1 = t_s[i, k].shape

    out = wishart_kl(t_eta, eta, t_s, s) + 0.5*D*torch.log(t_l/l) + 0.5*D*l/t_l - 0.5*l*temp4

    return out


def z_niw(D, eta, logdetS):
    """

    :param D:
    :param eta:
    :param logdetS:
    :return:
    """

    s = -0.5*eta*logdetS
    for d in range(1, D+1):
        s += torch.lgamma(0.5*(eta + 1 - d))

    return s

def digamma_sum(D, eta, logdetS):
    """
    The sum of digammas term defined in the appendix.

    :param D: the dimensionality of the data
    :param eta: the eta parameter
    :param logdetS: the log det of the S parameter
    :return:
    """
    s = -logdetS
    for d in range(1, D + 1):
        s += torch.digamma(0.5*(eta + d - D))

    return s


def wishart_kl(a_q, a_p, B_q, B_p):
    """
    The Kl for the wishart distribution
    :param a_q: posterior a parameter
    :param a_p: prior a parameter
    :param B_q: posterior B parameter
    :param B_p: prior B parameter
    :return:
    """

    N, K, D, _ = B_p.shape

    DBq = torch.zeros([N, K])
    DBp = torch.zeros([N, K])
    trc = torch.zeros([N, K])
    det = torch.zeros([N, K])
    for i in range(N):
        for k in range(K):
            DBq[i, k] = torch.logdet(B_q[i, k]/2.)
            DBp[i, k] = torch.logdet(B_p[i, k]/2.)
            DBqp = torch.mm( B_p[i, k],  torch.inverse(B_q[i, k]))
            trc[i, k] = torch.trace(DBqp)
            det[i, k] = torch.logdet(DBqp)

    z_prior = z_niw(D, a_p, DBp)
    z_post = z_niw(D, a_q, DBq)
    trc_term = 0.5 * a_q * (trc - D)
    digamma_term = 0.5*(a_q - a_p)*digamma_sum(D, a_q, DBq)

    return z_prior - z_post + digamma_term + trc_term


def _batch_diag(bmat):
    r"""
    Returns the diagonals of a batch of square matrices.
    """
    return torch.diagonal(bmat, dim1=-2, dim2=-1)


def _batch_trtrs_lower(bb, bA):
    """
    Applies `torch.trtrs` for batches of matrices. `bb` and `bA` should have
    the same batch shape.

    code from the torch git.
    """
    flat_b = bb.reshape((-1,) + bb.shape[-2:])
    flat_A = bA.reshape((-1,) + bA.shape[-2:])
    flat_X = torch.stack([torch.trtrs(b, A, upper=False)[0] for b, A in zip(flat_b, flat_A)])
    return flat_X.reshape(bb.shape)


def _batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.

    Code from the torch git.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)


def batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.

    This code was taken from the pytorch git.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.triangular_solve(flat_x_swap, flat_L, upper=False)[0].pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)
