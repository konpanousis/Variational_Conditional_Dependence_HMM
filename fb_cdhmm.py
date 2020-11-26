"""
File containing the computations for the forward-backward variant as derived in the Variational Conditional Dependence
Hidden Markov Models.

@author: Anonymous
"""

import torch
import sys
import numpy as np

DTYPE = torch.float


def fb_cdhmm(log_prob, K, N, cur_log_pi, cur_log_A, cur_log_hat_A, backward = True):
    """
    Forward backward variant derived for the cdhmm model.

    :param log_prob: the log probability for the current action sequence. TxN
    :param K: the number of states for the postulated first layer process, int
    :param N: the number of observation emitting states, int
    :param cur_log_pi: expected log value of pi, N
    :param cur_log_A: expected log value of A, KxNxN
    :param cur_log_hat_A: expected log A of hat A, KxK
    :param backward: flag to compute the backward messages, boolean
    :return: alpha, beta and the responsibilities
    """

    # get alpha, beta (if flag else zeros) and the log likelihoods
    alpha, beta, llh = fwbw(log_prob, K, N, cur_log_pi, cur_log_A, cur_log_hat_A, backward = backward)

    # if backward flag, compute responsibilities
    if backward:
        gamma_z, gamma_q, trans_z, trans_q = _calculate_gammas(log_prob, K, N, alpha, beta, cur_log_A, cur_log_hat_A)

    else:
        gamma_z = 0.
        gamma_q = 0.
        trans_z = 0.
        trans_q = 0.

    return alpha, beta, gamma_z, gamma_q, trans_z, trans_q, llh


def fwbw(log_prob, K, N, cur_log_pi, cur_log_A, cur_log_hat_A, backward=True):
    """
    The main function for computing foward backward variant derived in CD-HMM model.

    :param log_prob: the log probability for the current action sequence, TxN
    :param K: the number of states for the postulated first layer process, int
    :param N: the number of observation emitting states, int
    :param cur_log_pi: expected log value of pi, N
    :param cur_log_A: expected log value of A, KxN,xN
    :param cur_log_hat_A: expected log A of hat A, NxN
    :param backward: flag to compute the backward messages, boolean
    :return: alpha, beta and log likelihoods
    """

    # get the num of time frames for the sequence
    T = log_prob.shape[0]

    # fix the shape for the messages, alphas and betas are T,K,N,N...
    shape = [T, K]
    shape_middle = [K, N]
    
    # some expansions
    for _ in range(K):
        shape.append(N)
        shape_middle.append(1)
    shape_middle = shape_middle[:-1]
    shape_middle[-1] = N

    # init the messages, we work in the log domain
    alpha = -np.inf * torch.ones(shape)
    beta = -np.inf * torch.ones(shape)

    # init alphas, pi is (1,N)
    # transform alpha_0 to be 1,K,N
    alpha[0, 0] = (cur_log_pi + log_prob[0, :])[ (None, ) * (K) + (...,)]

    # compute the forward messages for each time frame
    for t in range(0, T - 1):
        # so alpha_t is K, N, N ... and cur_log_hat_A is K, N, N. need to expand to the inner dots
        temp = torch.logsumexp(cur_log_A.reshape(shape_middle) + alpha[t], -1, keepdim = True).unsqueeze(1) # K, 1, N, N,...
        alpha[t + 1] = torch.logsumexp(temp + cur_log_hat_A.t()[(...,) + (None, )*(K)], 0) # K, N, N, ...
        alpha[t + 1] += log_prob[t + 1, :][(None, )*(K) + (...,)]

        # set impossible states to -inf
        if t+2<K:
            alpha[t+1, K-t-2:K-t+1, ...] = -np.inf

    # analogous logic for the backward computation
    if backward:
        beta[-1] = torch.zeros_like(beta[-1])

        for t in range(T-2, -1, -1):
            # cur log hat pi is K,K , cur log pi is K, N,N ,beta_t is K, N, N,..., obs is N (the last axis)
            # make all K,K, N, ...., logsumexp q_t+1 and then logsumexp z_t+1
            temp = torch.logsumexp( log_prob[t+1][(None,)*K + (...,)] + beta[t+1] +
                                    cur_log_A.reshape(shape_middle), -1, keepdim = True).unsqueeze(1) # K, 1, N,...
            beta[t] = torch.logsumexp(temp + cur_log_hat_A[(...,) + (None, ) *K], 0)

            # set some impossible states to inf
            if K -t >0:
                beta[t, t+1:, ...] = -np.inf

    ll = alpha[-1]
    while ll.shape != ():
        ll = torch.logsumexp(ll, -1)
    return alpha, beta, ll


#######################################
######### Sufficient Statistics #######
#######################################
def _calculate_gammas(log_prob, K, N, alpha, beta, cur_log_A, cur_log_hat_A):
    """
    Compute the necessary responsibilities that comprise the variational posterior of CD-HMM.

    :param log_prob: the log probability for the current sequence
    :param K: the number of states for the postulated first layer process, int
    :param N: the number of observation emitting states, int
    :param cur_log_pi: expected log value of pi, N
    :param alpha: the alpha messages
    :param beta: the beta messages
    :param cur_log_A: expected log value of A, KxN,xN
    :param cur_log_hat_A: expected log A of hat A, NxN
    :return: the responsibilities
    """

    # get the time frames for the current sequence
    T = log_prob.shape[0]

    # fix the shape for the messages, alphas and betas are T,K,N,N...
    shape = [T, K]
    shape_middle = [K, 1, N]

    # find the shapes for the responsibilities
    for _ in range(K):
        shape_middle.append(1)
    shape_middle = shape_middle[:-1]
    shape_middle[-1] = N

    # we reduce over all dimensions except last one and the first one to calc gammaz and gammaq
    gamma_reduction_shapes = [2]*(K-1)

    # first and second level marginal states
    gamma_ = alpha + beta
    for red in gamma_reduction_shapes:
        gamma_ = torch.logsumexp(gamma_, red)

    gamma_q = torch.logsumexp(gamma_, 1)
    gamma_z = torch.logsumexp(gamma_, -1)

    assert gamma_q.shape == (T, N)
    assert gamma_z.shape == (T, K)

    # transitions temps
    # init some variables for easier calculations
    # this might be big, so maybe change it later
    # will reduce over t in each iter
    size = [K, K]
    size.extend([N]*K)

    # reminder beta_t is K,N,..., alpha_t, is K,N...., cur_log_A is K, N, N, cur_log_hat_A is K,K and obs T,N
    reduction_shape_z = [2]*K
    reduction_shape_q = [1]*1 + [2]*(K-2)
    temp_z = torch.zeros([T, K, K])
    temp_q = torch.zeros([T, K, N, N])

    for t in range(1, T):
        tempz = beta[t].unsqueeze(0) + log_prob[t][(None,)*(K+1) + (...,)] + cur_log_A.reshape(shape_middle) + alpha[t-1].unsqueeze(1)
        for dim in reduction_shape_z:
            tempz = torch.logsumexp(tempz, dim)
        temp_z[t] = tempz

        tempq = beta[t].unsqueeze(0) + cur_log_hat_A[(...,) + (None, ) * K] + cur_log_A.reshape(shape_middle) + alpha[t-1].unsqueeze(1)

        for dim in reduction_shape_q:
            tempq = torch.logsumexp(tempq, dim)
        temp_q[t] += tempq + log_prob[t][None, :, None]

    trans_z = temp_z + cur_log_hat_A[None, :, :]
    trans_z -= torch.logsumexp(trans_z, -1, keepdim = True)
    trans_z = torch.where(torch.isfinite(trans_z), trans_z, -np.inf*torch.ones_like(trans_z))
    trans_z = torch.logsumexp(trans_z, 0)

    trans_q = temp_q
    trans_q -= torch.logsumexp(trans_q, -1, keepdim = True)
    trans_q = torch.where(torch.isfinite(trans_q), trans_q, torch.ones_like(trans_q))
    trans_q = torch.logsumexp(trans_q, 0)

    gamma_z -= torch.logsumexp(gamma_q, -1, keepdim= True)
    gamma_q -= torch.logsumexp(gamma_q, -1, keepdim = True)

    return torch.exp(gamma_z), torch.exp(gamma_q), torch.exp(trans_z), torch.exp(trans_q)
