"""
Utility functions
"""
import hashlib
import os
import pickle as pkl

import numpy as np

__author__ = "Daheng Wang"
__email__ = "dwang8@nd.edu"


class AliasMethod:
    """
    See https://en.wikipedia.org/wiki/Alias_method
    """
    def __init__(self, prob_lst):
        self.n = len(prob_lst)
        self.U = []
        self.K = []

        prob_lst_md5 = fast_md5_lst(prob_lst)
        cache_file = os.path.join('.', '{}.alias.pkl'.format(prob_lst_md5))

        if os.path.exists(cache_file):
            '''If cache file for list U and K exist, load and cut'''
            print('Cached sampler load!', end='\t', flush=True)
            with open(cache_file, 'rb') as f:
                u_k_lst = pkl.load(f)
            self.U = u_k_lst[:int(len(u_k_lst)/2)]
            self.K = u_k_lst[int(len(u_k_lst)/2):]
        else:
            '''If no cache file, compute U and K. Cache U + K list at last'''
            print('Building and caching...', end='\t', flush=True)
            self.K = [i for i in range(self.n)]

            overfull = []
            underfull = []
            exactfull = []
            for i in range(self.n):
                u_i = self.n * prob_lst[i]
                self.U.append(u_i)
                if u_i > 1:
                    overfull.append(i)
                elif u_i < 1:
                    underfull.append(i)
                else:
                    pass

            while len(overfull) and len(underfull):
                i = overfull.pop()
                j = underfull.pop()
                self.K[j] = i
                self.U[i] = self.U[i] + self.U[j] - 1
                exactfull.append(j)
                if self.U[i] > 1:
                    overfull.append(i)
                elif self.U[i] < 1:
                    underfull.append(i)
                else:
                    pass
            '''Write to cache pickle file'''
            with open(cache_file, 'wb') as f:
                u_k_lst = self.U + self.K
                pkl.dump(u_k_lst, f)

    def sample(self):
        x = np.random.rand()
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)

        if y < self.U[i]:
            return i
        else:
            return self.K[i]


def fast_md5_lst(long_lst, cut_len=7):
    """
    Fast generate md5 hash value of a long list
    """
    target_lst = long_lst
    if len(long_lst) > 10**cut_len:  # If more than 1 million elements in list
        target_lst = long_lst[:10**cut_len]
    p = pkl.dumps(target_lst, -1)
    target_lst_md5 = hashlib.md5(p).hexdigest()
    return target_lst_md5
