"""
Script file for running the hierarchical HMM extension introduced in Variational Conditional Dependence
Hidden Markov Models for Human Action Recognition.

@author: Anonymous
"""
import torch
import torch.nn.functional as F

import numpy as np
import time
import os, sys
from data import load_action_ready
from utils import *
import math
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy import io
from sklearn.metrics import confusion_matrix
from scipy import io as sio
from fb_cdhmm import fb_cdhmm
from sklearn import preprocessing

DTYPE = torch.float
save_iters = 1


# set random seed
#torch.manual_seed(0)
#np.random.seed(0)

#############################
#### VB CDHMM CLASS ########
#############################
class CDHMM(torch.nn.Module):
    """
    Class CDHMM containing all necessary, inits, functions and calls to train a vb-cd-hmm model.
    """

    def __init__(self, num_seq, D, K, N, M, cov_prior_fact = 0.1, save_path = '', params = {}):
        """
        Init the CDHMM model.
        Create all the necessary variables.
        :param num_seq: the number of sequences
        :param D: the dimension of the data
        :param K: the states of the first postulated process
        :param N: the emission states
        :param M: the number of mixtures
        :param cov_prior_fact: a factor to add to the covariance to avoid concentrated results.a
        :param save_path: the path to save results
        """
        super().__init__()

        self.num_seq = num_seq

        self.M = M
        self.N = N
        self.K = K
        self.D = D

        ###############
        ## PRIORS #####
        ###############s

        # hyperparameters for initial state and transitions of the first layer process
        self.eta_0 = torch.nn.Parameter(1e-3 * torch.tensor(np.ones([N], dtype = np.float32)), requires_grad = False)
        self.eta = torch.nn.Parameter(1e-3 * torch.ones([K, N, N], dtype = DTYPE), requires_grad = False)

        self.eta_hat_0 = torch.nn.Parameter(1e-3*torch.ones([K])  , requires_grad=False)
        self.eta_hat = torch.nn.Parameter(1e-3 * torch.ones([K,K])  , requires_grad=False)

        self.l = torch.nn.Parameter(0.25*torch.ones([N, M], dtype = DTYPE), requires_grad = False)
        self.eta_nw = torch.nn.Parameter((D + 2.) * torch.ones([N, M], dtype = DTYPE), requires_grad = False)
        self.w = torch.nn.Parameter(1e-3 * torch.ones([N, M], dtype = DTYPE), requires_grad = False)

        ##################
        # POSTERIORS
        ##################
        self.m = torch.nn.Parameter(torch.tensor(params['mu0']).unsqueeze(0).expand([N, M, D]).float(),
                                    requires_grad=False)
        self.S = torch.nn.Parameter(torch.tensor(params['sigma0']).unsqueeze(0).expand([N, M, D, D]).float(),
                                    requires_grad=False)

        self.cov_prior = cov_prior_fact* torch.eye(D)[None,None,:,:].expand([N, M, D, D]).float()
        self.S.data += self.cov_prior

        # some extra params for the NIW distro
        self.t_l = torch.nn.Parameter(.25 * torch.ones([N, M], dtype = DTYPE), requires_grad = False)
        self.t_m = torch.nn.Parameter(torch.normal(mean = torch.tensor(params['mu']).float(), std = .01).float(), requires_grad = False)
        self.t_eta_nw = torch.nn.Parameter((D + 2.) * torch.ones([N, M], dtype = DTYPE), requires_grad = False)
        self.t_S = torch.nn.Parameter(torch.tensor(params['sigma']).float(), requires_grad = False)
        self.t_S = torch.nn.Parameter(torch.eye(D)[None,None,:,:].expand([N,M,D,D]).float().clone(), requires_grad = False)
        self.t_w = torch.nn.Parameter(1e-3*torch.ones_like(torch.tensor(params['w'])).float(), requires_grad = False)

        # parameters for the second layer process
        t_A = torch.tensor(params['transmat']).float().unsqueeze(0).expand([K, N, N])
        self.t_A = torch.nn.Parameter((t_A), requires_grad = False)
        self.t_pi = torch.nn.Parameter((torch.tensor(params['init_state'])).float(), requires_grad = False)

        # params for the postulated first layer process
        t_hat_A = (1e-1 - 1e-3) * torch.rand([K,K]) + 1e1
        t_hat_pi = (1e-1 - 1e-3) * torch.rand([K]) + 1e1
        t_hat_A = torch.ones([K,K])
        t_hat_pi = torch.ones([K])
        self.t_hat_A = torch.nn.Parameter(t_hat_A, requires_grad=False)
        self.t_hat_pi = torch.nn.Parameter(t_hat_pi, requires_grad=False)

        self.priors = [self.eta_0, self.eta, self.l, self.m, self.eta_nw, self.S, self.w]
        self.params = [self.t_l, self.t_m, self.t_eta_nw, self.t_S, self.t_w, self.t_A, self.t_pi]

        self.save_path = save_path


    def fit(self, X,  n_iter_em = 1, cur_run = -1, cur_act = -1, verbose = False):
        """
        Fit the hmm to the data X
        :param X: the data, a ndarray object with size (num_seqs,)
        :param n_iter_em: the number of em iterations to perform
        :param cur_run:  the cur run (in case of multiple runs
        :param cur_act: the current action
        :param verbose: print some stuff
        :return: none, updates parameters, assigns self objects, etc
        """

        self.log_probs = []
        cur_log_prob = 0.

        # run EM
        for iter in range(n_iter_em):

            start = time.time()

            if verbose:
                if (iter + 1) % 1 == 0:
                    print("Cur act:", cur_act, "cur run:", cur_run, "iter:", iter, "log prob",
                          cur_log_prob)

            stats = self._init_suff_stats()
            cur_log_prob = 0.

            # set the expected values for the random variables
            self.set_expectations()

            # parse all the sequences for this action
            for i in range(len(X)):

                cur_x = torch.from_numpy(X[i].T).float()

                # compute the expected log likelihood
                self._log_prob = self._compute_log_likelihood(cur_x, normalise =  False)

                # calculate the log likelihoods and the responsibilities for updating the params
                ll, _, gamma_z, gamma_q, trans_z, trans_q, llh = \
                    fb_cdhmm(self._log_prob, self.K, self.N, self.cur_log_pi, self.cur_log_A, self.cur_log_hat_A)
                self.acc_suff_stats(stats, gamma_z, gamma_q, trans_z, trans_q, cur_x)

                ll = ll[-1]
                while ll.shape != ():
                    ll = torch.logsumexp(ll, -1)
                cur_log_prob += ll/len(cur_x)

            # maximization steps
            self._state_maximization_step(stats)
            self._global_maximization_step(stats)

            #print('EM step time:', time.time() - start)

            # get current log prob for all sequences
            cur_log_prob = self._vlb(X, cur_log_prob, kl = True, verbose = verbose)

            # check convergence and if true break
            converged, save = converge(self.log_probs, cur_log_prob)

            # save the state dictionary
            if save:
                torch.save(self.state_dict(), self.save_path)

            # break on convergence
            if converged:
                break

            self.elbo = cur_log_prob




    def _vlb(self, X, cur_log_prob, kl = True, verbose = False):
        """
        Evidence Lower bound calculation.
        :param X: the current sequence of data
        :param cur_log_prob: the cur log_probability (computed using fb variant)
        :param kl: compute KL?, boolean
        :param verbose: flag, if true print some stuff for debugging
        :return:
        """

        final_prob = cur_log_prob.clone()

        if kl:

            # the KL divergence for the dirichlet priors
            varpi_kl = dirichlet_kl(self.eta_0, self.t_pi).sum()
            pi_kl = dirichlet_kl(self.eta, self.t_A).sum()

            varpi_hat_kl = dirichlet_kl(self.eta_hat_0, self.t_hat_pi).sum()
            pi_hat_kl = dirichlet_kl(self.eta_hat, self.t_hat_A).sum()

            w_kl = dirichlet_kl(self.w, self.t_w).sum()

            # and for the NIW
            rest_kl = niw_kl(self.t_m, self.m, self.t_l, self.l, self.t_eta_nw, self.eta_nw, \
                                   self.t_S, self.S).sum()

            final_prob -= varpi_kl
            final_prob -= pi_kl

            final_prob -= varpi_hat_kl
            final_prob -= pi_hat_kl

            final_prob -= w_kl
            final_prob -= rest_kl

            if verbose:
                print ("{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, "
                       "All: {5:.2f}".format(cur_log_prob, varpi_kl, pi_kl, w_kl, rest_kl, final_prob))
        else:
            print(final_prob)

        return final_prob


    def get_expected_log_prob(self, l, m, eta, S, y):
        """
        Helper function for computing the expected log likelihood (see text for the params) and the calculation of
        the expected log likelihood.
        :param l: lambda parameter
        :param m: m parameter
        :param eta: eta parameter
        :param S: S parameter
        :param y: the current sequence
        :return: the expected log prob for the given sequence and state parameters
        """

        y = y.unsqueeze(-2)

        try:
            chol = torch.cholesky(S)
        except RuntimeError:
            #print(S)
            chol = torch.cholesky(
                S + 1e-3 * torch.eye(self.D).unsqueeze(0).expand(self.M, self.D, self.D).float())

        # compute the mahalanobis distance
        diff = y - m
        M = batch_mahalanobis(chol, diff)

        # this is half log det
        half_log_det = chol.diagonal(dim1 = -2, dim2 = -1).log().sum(-1)

        exp_log_R_2 = torch.sum(torch.digamma(0.5 * (eta.unsqueeze(0) + 1. -torch.arange(1., self.D + 1.).unsqueeze(-1))),
                                -2)
        exp_log_R_2 = torch.where(torch.isfinite(exp_log_R_2), exp_log_R_2, 0. * torch.ones_like(exp_log_R_2))

        term_1 = exp_log_R_2 + self.D * torch.log(torch.tensor(2.)) + 2. * half_log_det

        return term_1 / 2. - self.D / 2. * torch.log(torch.tensor(2. * math.pi)) - self.D / (2. * l) - (eta / 2.) * M


    def _compute_log_likelihood(self, y, reduce = True, normalise = False):
        """
        Compute the expected log likelihood of the data.

        :param y: the current sequence/
        :param reduce: reduce over the mixture components, boolean
        :return: the reduced or not expected log likelihoods
        """

        _log_likelihoods = torch.zeros([self.N, y.shape[0], self.M])

        for n in range(self.N):

            # get the params for the specific state
            cur_l = torch.squeeze(self.t_l[n])
            cur_m = torch.squeeze(self.t_m[n])
            cur_eta = torch.squeeze(self.t_eta_nw[n])
            cur_S = torch.squeeze(self.t_S[n])
            cur_w = torch.squeeze(self.t_w[n].clone())

            log_expected_weights = expectation_log_dirichlet(cur_w)

            if normalise:
                log_expected_weights -= torch.logsumexp(log_expected_weights, -1, keepdim = True)

            # some checks
            if not torch.all(torch.isfinite(log_expected_weights)):
                print(self.t_w)
                print('Oops, non finite log expected weights.')
                sys.exit()

            _log_likelihoods[n] = self.get_expected_log_prob(cur_l, cur_m, cur_eta, cur_S, y)
            _log_likelihoods[n] = torch.where(torch.isfinite(_log_likelihoods[n]), \
                                              _log_likelihoods[n], torch.ones_like(_log_likelihoods[n]))
            _log_likelihoods[n] += log_expected_weights

        if reduce:
            return torch.logsumexp(_log_likelihoods, -1).t()
        else:
            perm = _log_likelihoods.permute([1, 0, 2])  # now it is T, K, M
            return perm

    def acc_suff_stats(self, stats, gamma_z, gamma_q, trans_z, trans_q, X):
        """
        Accumulate sufficient statistics after the E step, in order to perform the updates
        :param stats: the dictionary of sufficient stats to update
        :param X: the current observation sequence
        :param gamma_z: the gamma z parameter, inital state responsibilities (see main text)
        :param gamma_q: the initial state responsibilities for the second layer process
        :param trans_z: the state transition responsibilities for the first layer process
        :param trans_q: the state transition responsibilities for the second layer process
        :return:
        """

        # accumulate for everything
        stats['init_state_all'] += gamma_q[0]
        stats['init_state_sec_all'] += gamma_z[0]

        # size of gamma ijt is T, K, N, N
        stats['transitions_all'] += trans_q
        stats['transitions_second_all'] += trans_z

        #  mixtures stats q(x_t, l_t)
        xi_log = self._compute_log_likelihood(X, reduce=False, normalise=False)
        norm = torch.logsumexp(xi_log, -1, keepdim=True)
        norm = torch.where(torch.isfinite(norm), norm, torch.zeros_like(norm))
        xi_log -= norm
        r = torch.exp(xi_log) * gamma_q.unsqueeze(-1)

        stats['emm_denom'] += torch.sum(r, 0)
        stats['r*obs'] += torch.einsum('ijk, il -> jkl', r, X)

        x_centered = X[:, None, None, :].expand(X.shape[0], self.N, self.M, self.D)  # - tm[None, :, :, :]

        centered = torch.reshape(x_centered, [X.shape[0], self.N, self.M, self.D, 1])
        centered_t = torch.reshape(x_centered, [X.shape[0], self.N, self.M, 1, self.D])
        centered_dot = r[:, :, :, None, None] * centered * centered_t

        stats['obs*r*obs'] += torch.sum(centered_dot, 0)


    #################################################
    ################## M STEP #######################
    #################################################
    def _state_maximization_step(self, stats, local = False, i = -1):
        """
        Update the state parameters for all layer processes. These include the initial state and transition probs.
        :param stats: the dictionary containing values of sufficient statistics
        :param local: if we want to use different params per sequence (to do)
        :param i: the current sequence (to do)
        """

        # update the initial states through gamma in (T,K)
        # note pi is in (N,K)
        # maybe add where all zeros back to uniform
        # *************

        if local:
            varpi_i = self.eta_0 + stats['init_state_all']
            self.t_pi[i].data = varpi_i.data

            pi_i = self.eta_0 + stats['transitions_all']
            self.t_A[i].data = pi_i.data
        else:

            varpi_i = self.eta_0 + stats['init_state_all']
            pi_i = self.eta + stats['transitions_all']
            hat_pi_i = self.eta_hat_0 + stats['init_state_sec_all']
            hat_A_i = self.eta_hat + stats['transitions_second_all']
            varpi_i /= varpi_i.sum(-1, keepdim = True)
            pi_i /= pi_i.sum(-1, keepdim = True)
            hat_pi_i /= hat_pi_i.sum(-1, keepdim = True)
            hat_A_i /= hat_A_i.sum(-1, keepdim = True)

            self.t_pi.data = varpi_i.data.clamp(1e-9, 10000.)
            self.t_A.data = pi_i.data.clamp(1e-9, 10000.)
            self.t_hat_pi.data = hat_pi_i.data.clamp(1e-9, 10000.)
            self.t_hat_A.data = hat_A_i.data.clamp(1e-9, 10000.)


    def _global_maximization_step(self, stats):
        """
        Update the emission distributions using the calculated responsibilities
        :param stats: the dictionary of responsibilities, normalizers, etc.
        :return: nothing, just update the parameters
        """

        t_gamma = stats['emm_denom']
        delta = stats['obs*r*obs']/(stats['emm_denom'][:,:, None, None] +1e-7)

        mm = stats['r*obs'][:,:,:,None]*stats['r*obs'][:,:,None, :]/(stats['emm_denom']**2 + 1e-7)[:,:,None,None]

        m0m0 = self.l.unsqueeze(-1).unsqueeze(-1)*self.m[:,:,:,None]*self.m[:,:, None, :]/(stats['emm_denom'][:,:,None,None] + 1e-3)

        self.t_m.data = (((self.l.unsqueeze(-1) * self.m) + stats['r*obs']) /
                         (self.l + t_gamma)[:, :,None]).data

        self.t_S.data = ( self.S + delta -mm + m0m0 +self.cov_prior.data).data

        self.t_eta_nw.data = (self.eta_nw + t_gamma).data.clamp(1e-3, 10000.)
        self.t_l.data = (self.l + t_gamma).data.clamp(1e-3, 10000.)
        self.t_w.data = (self.w + t_gamma).data.clamp(1e-3, 10000.)

        if torch.any(torch.isnan(self.t_S)):
            print('nan t_s')
            print(delta.max())
            print(t_gamma.max())
            sys.exit()

        if torch.any(torch.isnan(self.t_w)):
            print('weights nan')
            print(self.t_w)
            sys.exit()


    ###############################
    ######### INFERENCE ###########
    ###############################
    def set_expectations(self):
        """
        Set the expectation of the respective random variables for the necessary computations.

        :return: none, just set the expectations
        """
        self.cur_log_pi = expectation_log_dirichlet(self.t_pi + 1e-20).unsqueeze(0)
        self.cur_log_pi = torch.where(torch.isfinite(self.cur_log_pi),
                                         self.cur_log_pi, -np.inf * torch.ones_like(self.cur_log_pi))

        self.cur_log_A = expectation_log_dirichlet(self.t_A + 1e-20)
        self.cur_log_A = torch.where(torch.isfinite(self.cur_log_A),
                                      self.cur_log_A, -np.inf * torch.ones_like(self.cur_log_A))

        self.cur_log_hat_A = expectation_log_dirichlet(self.t_hat_A + 1e-20)
        self.cur_log_hat_A = torch.where(torch.isfinite(self.cur_log_hat_A),
                                          self.cur_log_hat_A, -np.inf * torch.ones_like(self.cur_log_hat_A))

        self.cur_log_hat_pi = expectation_log_dirichlet(self.t_hat_pi + 1e-20).unsqueeze(0)
        self.cur_log_hat_pi = torch.where(torch.isfinite(self.cur_log_hat_pi),
                                             self.cur_log_hat_pi, -np.inf * torch.ones_like(self.cur_log_hat_pi))



    def inference(self, X):
        """
        Use the forward backward variant for the predictive density. Described in the main text.
        :param X: the input
        :return:
        """

        self._log_prob = self._compute_log_likelihood(X, normalise = False)
        alpha = fb_cdhmm(self._log_prob, self.K, self.N, self.cur_log_pi, self.cur_log_A,
                       self.cur_log_hat_A, backward=False)[0]
        logtot = alpha[-1]

        # reduce until we get a scalar
        while logtot.shape != ():
            logtot = torch.logsumexp(logtot, -1)

        return logtot

    ########################################################################
    ############################## SUFFICIENT ##############################
    ########################################################################
    def _init_suff_stats(self):
        """
        Initialize the dictionary to gather the sufficient statistics for every sequence.
        :return: the defined dictionary
        """

        stats = {
            'init_state_all': torch.zeros([self.N], dtype=DTYPE),
            'init_state_sec_all': torch.zeros([self.K], dtype=DTYPE),
            'transitions_all': torch.zeros((self.K, self.N, self.N), dtype=DTYPE),
            'transitions_second_all': torch.zeros([self.K, self.K], dtype=DTYPE),
            'r*obs*obs.T': torch.zeros([self.N, self.M, self.D, self.D], dtype=DTYPE),
            'tilde_x': torch.zeros([self.N, self.M, self.D], dtype=DTYPE),
            'tilde_denom': torch.zeros([self.N, self.M, self.D], dtype=DTYPE),
            'r*obs': torch.zeros([self.N, self.M, self.D], dtype=DTYPE),
            'emm_denom': torch.zeros([self.N, self.M]),
            'obs*r*obs': torch.zeros([self.N, self.M, self.D, self.D])
        }

        return stats



def test_cdhmm_skeleton():
    """
    Train and test the model.
    :return:
    """
    # different configurations
    # M number of mixtures, K number of states, N number of emission states
    M = [2,3, 4, 5,6,7,8]
    N = [10,11,12,13,14,15]
    K = [2]

    best_acc = 0.
    accs_dict = {}

    # some user input
    dataset_num = int(input('Please choose the dataset:\n 1. UTD \n 2. G3D \n 3. MSRA \n 4. PENN\n\nInput:'))

    # portion of missing values
    portion = '0'

    # choose dataset according to user input
    if dataset_num == 1:
        dataset = 'utd'
        n_actions = 27
        cov_prior_reg = 0.01
    elif dataset_num == 2:
        dataset = 'g3d'
        n_actions = 20
        cov_prior_reg = 0.01
    elif dataset_num == 3:
        dataset = 'msra'
        n_actions = 20
        cov_prior_reg = 0.01
    elif dataset_num == 4:
        dataset = 'penn'
        n_actions = 11
        cov_prior_reg = 100.
    else:
        print('Wrong dataset num. Please Try Again..')
        sys.exit()

    print('Dataset...'+ dataset)
    train = "1"

    if train == '1':
        n_iter_em = int(input('Number of EM iterations: '))
        n_runs = int(input('Number of runs per action: '))
    else:
        n_iter_em = 1
        n_runs = 1

    actions = range(1,n_actions + 1)

    x_train_all = np.array([])
    x_test_all = np.array([])
    labels = []

    # read all the data
    for i in range(1, n_actions + 1):

        X_tr, X_te = load_action_ready(i, dataset = dataset, portion = portion)

        x_train_all = np.hstack([x_train_all, X_tr])
        if i<= len(actions):
            x_test_all = np.hstack([x_test_all, X_te])
            labels.extend([float(i)] * len(X_te))

    print('###### DATASET STATS ##########')
    print('## Num actions:', n_actions, '#######')
    print('## Times all:', x_train_all.shape, '######')
    print('## Dimensionality:', X_tr[0].shape[0], '######')
    print('###################################')


    # train and test combined
    if train == '1':

        # create different dirs for different experiments
        if portion:
            configuration_path = "./configurations_missing"
        else:
            configuration_path = "./configurations"
        if not os.path.exists(configuration_path):
            os.mkdir(configuration_path)
        configuration_path += '/' + str(dataset)
        if not os.path.exists(configuration_path):
            os.mkdir(configuration_path)
        else:
            count = 1
            new_path = configuration_path + '_' + str(count)
            while True:
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                    break
                count += 1
                if count>=10:
                    new_path = new_path[:-2] + str(count)
                else:
                    new_path = new_path[:-1] + str(count)
            configuration_path = new_path

        if portion:
            configuration_path += "/missing_" + portion
            if not os.path.exists(configuration_path):
                os.mkdir(configuration_path)

        # run muktiple configurations
        for iter in range(n_runs):
            for n_jumps in K:
                for n_mix in M:
                    for n_states in N:

                        print('Iteration:', iter, ' N_states:', n_states, ' N_mix:', n_mix, ' N_jumps:', n_jumps)

                        cur_configuration_path = configuration_path + "/n_states_" + str(n_states) +'_n_mix_'+ str(n_mix) +\
                        '_n_jumps_' + str(n_jumps) + '_run_' +str(iter) + '_n_iters_'+ str(n_iter_em) + '/'

                        if not os.path.exists(cur_configuration_path):
                            os.mkdir(cur_configuration_path)

                        # initialize the likelihoods to zeros and the prior means and sigmas
                        log_likelihoods = np.zeros([x_test_all.shape[0], len(actions)])
                        mu0, sigma0, _ = mixgauss_init(np.hstack(x_train_all), n_mix)

                        # run configurations for each action
                        for acts in range(len(actions)):

                            act = actions[acts]

                            SAVE_PATH = cur_configuration_path + 'cdhmm-action-' + str(act)
                            if not os.path.exists(SAVE_PATH):
                                os.mkdir(SAVE_PATH)

                            SAVE_PATH = SAVE_PATH + '/state_dict.pt'

                            init_time = time.time()

                            # load the current action for the given dataset
                            X_train, _ = load_action_ready(act, dataset = dataset, portion = portion)


                            print('Action: ', act)

                            # init parameters
                            prior, transmat, _, _, mu, sigma, w, _ = init(X_train, n_states, n_mix, X_train[0].shape[0])

                            params = {'mu0': mu0, 'sigma0': sigma0, 'init_state': prior, 'transmat': transmat, 'mu': mu,
                                      'sigma': sigma, 'w': w}

                            # create a new instance
                            hmm = CDHMM(len(X_train), X_train[0].shape[0], n_jumps, n_states, n_mix, cov_prior_reg, save_path = SAVE_PATH,
                                         params = params)

                            # fit the model
                            hmm.fit(X_train, n_iter_em = n_iter_em, cur_run = iter, cur_act = act)

                            print('Action:' + str(act) + ' time: ' + str(time.time() - init_time), end=' ')

                            # get likelihoods for all test data for this trained model
                            init_time = time.time()
                            for x_test_ind in range(len(x_test_all)):
                                ll = hmm.inference(torch.tensor(x_test_all[x_test_ind].T).float())
                                log_likelihoods[x_test_ind, acts] = ll.numpy()
                            print('Inference Time:', time.time() - init_time)
                            torch.save(hmm.state_dict(), SAVE_PATH)

                        # choose the label for each test sequence using the log probabilities
                        pred = (np.argmax(log_likelihoods, -1) + 1.).astype( np.float32)

                        acc = np.mean(np.equal(pred, labels))
                        print('Accuracy: ', acc)
                        print('\n\n')

                        # get the confusion matrix
                        cm = confusion_matrix(labels, pred)
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                        # write some stuff to assess results
                        with open(cur_configuration_path+'accuracy.txt', 'w') as f:
                            f.write("Accuracy: "+ str(acc))
                        np.savetxt(cur_configuration_path+'confusion_matrix.out', cm)

                        accs_dict["N_"+str(n_states)+"_K_"+str(n_jumps)+"_M_"+str(n_mix)+'_run_'+str(iter)] = acc
                        with open(configuration_path+"/accs_dict_final.txt", 'w') as f:
                            for key in accs_dict:
                                f.write(key+" "+str(accs_dict[key]))
                                f.write('\n')

                        # write best acc to different file
                        if acc > best_acc:
                            best_acc = acc
                            best_conf = [n_states, n_jumps, n_mix]
                            with open(configuration_path + "/best_accs_final.txt", 'w') as f:
                                f.write(str(best_acc))
                                f.write('\n')
                                f.write(str(best_conf))


if __name__ == '__main__':
    test_cdhmm_skeleton()
