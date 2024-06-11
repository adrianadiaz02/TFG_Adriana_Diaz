#!/usr/bin/env python

"""Module with class for single video"""

__author__ = 'Anna Kukleva (base code), Adriana Díaz Soley (modifications)'
__date__ = 'August 2018, modified in May 2024'

from collections import Counter
import numpy as np
import math as m
import os
from os.path import join
from ute.utils.logging_setup import logger
from ute.utils.util_functions import dir_check
from ute.viterbi_utils.viterbi import Viterbi
from ute.viterbi_utils.grammar import Grammar
import itertools

import torch
import torch.nn.functional as F



class Video(object):
    """Single video with respective for the algorithm parameters"""
    def __init__(self, path, K, *, opt, gt=[], name='', start=0, with_bg=False):
        """
        Args:
            path (str): path to video representation
            K (int): number of subactivities in current video collection
            reset (bool): necessity of holding features in each instance
            gt (arr): ground truth labels
            gt_with_0 (arr): ground truth labels with SIL (0) label
            name (str): short name without any extension
            start (int): start index in mask for the whole video collection
        """
        self.iter = 0
        self.path = path
        self._K = K
        self.name = name
        self.opt = opt

        self._likelihood_grid = None
        self._valid_likelihood = None
        self._theta_0 = 0.1
        self._subact_i_mask = np.eye(self._K)
        self.n_frames = 0
        self._features = None
        self.global_start = start
        self.global_range = None

        self.gt = gt
        self._gt_unique = np.unique(self.gt)

        self.features()
        self._check_gt()

        # counting of subactivities np.array
        self.a = np.zeros(self._K)
        # ordering, init with canonical ordering
        self._pi = list(range(self._K))
        self.inv_count_v = np.zeros(self._K - 1)
        # subactivity per frame
        self._z = []
        self._z_idx = []
        self._init_z_framewise()

        # temporal labels
        self.temp = None
        self._init_temporal_labels()

        # background
        self._with_bg = with_bg
        self.fg_mask = np.ones(self.n_frames, dtype=bool)
        if self._with_bg:
            self._init_fg_mask()

        self._subact_count_update()

        self.segmentation = {'gt': (self.gt, None)}
        

    def features(self):
        """Load features given path if haven't do it before"""
        if self._features is None:
            if self.opt.ext == 'npy':
                self._features = np.load(self.path)
            else:
                self._features = np.loadtxt(self.path)
            ######################################
            # fixed.order._coffee_mlp_!pose_full_vae0_time10.0_epochs60_embed20_n1_!ordering_gmm1_one_!gt_lr0.0001_lr_zeros_b0_v1_l0_c1_.pth.tar

            # if self._features.shape[-1] == 65 and self.opt.feature_dim == 64:
            #     self._features = self._features[1:, 1:]
            #     np.savetxt(self.path, self._features)

            # if self.opt.data_type == 0 and self.opt.dataset == 'fs':
            #     self._features = self._features.T

            if self.opt.f_norm:  # normalize features
                mask = np.ones(self._features.shape[0], dtype=bool)
                for rdx, row in enumerate(self._features):
                    if np.sum(row) == 0:
                        mask[rdx] = False
                z = self._features[mask] - np.mean(self._features[mask], axis=0)
                z = z / np.std(self._features[mask], axis=0)
                self._features = np.zeros(self._features.shape)
                self._features[mask] = z
                self._features = np.nan_to_num(self.features())

            self.n_frames = self._features.shape[0]
            self._likelihood_grid = np.zeros((self.n_frames, self._K))
            self._valid_likelihood = np.zeros((self.n_frames, self._K), dtype=bool)
        return self._features

    def _check_gt(self):
        try:
            assert len(self.gt) == self.n_frames
        except AssertionError:
            # print(self.path, '# gt and # frames does not match %d / %d' % (len(self.gt), self.n_frames))
            if abs(len(self.gt) - self.n_frames) > 50:
                if self.opt.data_type == 4:
                    os.remove(os.path.join(self.opt.gt, self.name))
                    os.remove(self.path)
                    try:
                        os.remove(os.path.join(self.opt.gt, 'mapping', 'gt%d%s.pkl' % (self.opt.frame_frequency, self.opt.gr_lev)))
                        os.remove(os.path.join(self.opt.gt, 'mapping', 'order%d%s.pkl' % (self.opt.frame_frequency, self.opt.gr_lev)))
                    except FileNotFoundError:
                        pass
                raise AssertionError
            else:
                min_n = min(len(self.gt), self.n_frames)
                self.gt = self.gt[:min_n]
                self.n_frames = min_n
                self._features = self._features[:min_n]
                self._likelihood_grid = np.zeros((self.n_frames, self._K))
                self._valid_likelihood = np.zeros((self.n_frames, self._K), dtype=bool)

    def _init_z_framewise(self):
        """Init subactivities uniformly among video frames"""
        # number of frames per activity
        step = m.ceil(self.n_frames / self._K)
        modulo = self.n_frames % self._K
        for action in range(self._K):
            # uniformly distribute remainder per actions if n_frames % K != 0
            self._z += [action] * (step - 1 * (modulo <= action) * (modulo != 0))
        self._z = np.asarray(self._z, dtype=int)
        try:
            assert len(self._z) == self.n_frames
        except AssertionError:
            logger.error('Wrong initialization for video %s', self.path)

    def _init_temporal_labels(self):
        self.temp = np.zeros(self.n_frames)
        for frame_idx in range(self.n_frames):
            self.temp[frame_idx] = frame_idx / self.n_frames
            # self.temp[frame_idx] = frame_idx

    def _init_fg_mask(self):
        indexes = [i for i in range(self.n_frames) if i % 2]
        self.fg_mask[indexes] = False
        # todo: have to check if it works correctly
        # since it just after initialization
        self._z[self.fg_mask == False] = -1

    def _subact_count_update(self):
        c = Counter(self._z)
        # logger.debug('%s: %s' % (self.name, str(c)))
        self.a = []
        for subaction in range(self._K):
            self.a += [c[subaction]]

    def update_indexes(self, total):
        self.global_range = np.zeros(total, dtype=bool)
        self.global_range[self.global_start: self.global_start + self.n_frames] = True

    def reset(self):
        """If features from here won't be in use anymore"""
        self._features = None

    def z(self, pi=None):
        """Construct z (framewise label assignments) from ordering and counting.
        Args:
            pi: order, if not given the current one is used
        Returns:
            constructed z out of indexes instead of actual subactivity labels
        """
        self._z = []
        self._z_idx = []
        if pi is None:
            pi = self._pi
        for idx, activity in enumerate(pi):
            self._z += [int(activity)] * self.a[int(activity)]
            self._z_idx += [idx] * self.a[int(activity)]
        if self.opt.bg:
            z = np.ones(self.n_frames, dtype=int) * -1
            z[self.fg_mask] = self._z
            self._z = z[:]
            z[self.fg_mask] = self._z_idx
            self._z_idx = z[:]
        assert len(self._z) == self.n_frames
        return np.asarray(self._z_idx)

    def update_z(self, z):
        self._z = np.asarray(z, dtype=int)
        self._subact_count_update()

    def likelihood_update(self, subact, scores, trh=None):
        # for all actions
        if subact == -1:
            self._likelihood_grid = scores
            if trh is not None:
                for trh_idx, single_trh in enumerate(trh):
                    self._valid_likelihood[:, trh_idx] = False
                    self._valid_likelihood[:, trh_idx] = scores[:, trh_idx] > single_trh
        else:
            # for all frames
            self._likelihood_grid[:, subact] = scores[:]
            if trh is not None:
                self._valid_likelihood[:, subact] = False
                self._valid_likelihood[:, subact] = scores > trh

        # Debugging statements
        """logger.debug(f"Video {self.name}: Likelihood Grid after update for subact {subact}")
        logger.debug(f"  Likelihood Grid Shape: {self._likelihood_grid.shape}")
        logger.debug(f"  Likelihood Grid Range: Min {self._likelihood_grid.min()}, Max {self._likelihood_grid.max()}")
        logger.debug(f"  Likelihood Grid Type: {self._likelihood_grid.dtype}")"""

    def valid_likelihood_update(self, trhs):
        for trh_idx, trh in enumerate(trhs):
            self._valid_likelihood[:, trh_idx] = False
            self._valid_likelihood[:, trh_idx] = self._likelihood_grid[:, trh_idx] > trh

    def save_likelihood(self):
        """Used for multiprocessing"""
        dir_check(os.path.join(self.opt.data, 'likelihood'))
        np.savetxt(os.path.join(self.opt.data, 'likelihood', self.name), self._likelihood_grid)
        # print(os.path.join(self.opt.data, 'likelihood', self.name))

    def load_likelihood(self):
        """Used for multiprocessing"""
        path_join = os.path.join(self.opt.data, 'likelihood', self.name)
        self._likelihood_grid = np.loadtxt(path_join)

    def get_likelihood(self):
        return self._likelihood_grid

    def _viterbi_inner(self, pi, save=False):
        grammar = Grammar(pi)
        if np.sum(self.fg_mask):
            viterbi = Viterbi(grammar=grammar, opt = self.opt, probs=(-1 * self._likelihood_grid[self.fg_mask]))
            viterbi.inference()
            viterbi.backward(strict=True)
            z = np.ones(self.n_frames, dtype=int) * -1
            z[self.fg_mask] = viterbi.alignment()
            score = viterbi.loglikelyhood()
        else:
            z = np.ones(self.n_frames, dtype=int) * -1
            score = -np.inf
        # viterbi.calc(z)
        if save:
            name = '%s_%s' % (str(pi), self.name)
            save_path = join(self.opt.output_dir, 'likelihood', name)
            with open(save_path, 'w') as f:
                # for pi_elem in pi:
                #     f.write('%d ' % pi_elem)
                # f.write('\n')
                f.write('%s\n' % str(score))
                for z_elem in z:
                    f.write('%d ' % z_elem)
        return z, score

    # @timing
    def viterbi(self, pi=None):
        if pi is None:
            pi = self._pi
        log_probs = self._likelihood_grid

        # Debugging statements
        """logger.debug("in video class")
        logger.debug(f"Viterbi Decoding for {self.name}:")
        logger.debug(f"  Initial Log Probs Shape: {log_probs.shape}")
        logger.debug(f"  Initial Log Probs Range: Min {log_probs.min()}, Max {log_probs.max()}")
        logger.debug(f"  Initial Log Probs Type: {log_probs.dtype}")"""
        
        if np.max(log_probs) > 0:
            self._likelihood_grid = log_probs - (2 * np.max(log_probs))
        
        #logger.debug(f"  Normalized Likelihood Grid Range: Min {self._likelihood_grid.min()}, Max {self._likelihood_grid.max()}")
        
        alignment, return_score = self._viterbi_inner(pi, save=True)
        self._z = np.asarray(alignment).copy()

        self._subact_count_update()

        name = str(self.name) + '_' + self.opt.log_str + 'iter%d' % self.iter + '.txt'
        np.savetxt(join(self.opt.output_dir, 'segmentation', name), np.asarray(self._z), fmt='%d')
        # print('path alignment:', join(self.opt.data, 'segmentation', name))
        #print(f"Viterbi decoding for video {self.name}: unique labels {np.unique(self._z)}")

        return return_score
    

    def update_fg_mask(self):
        self.fg_mask = np.sum(self._valid_likelihood, axis=1) > 0

    def resume(self):
        name = str(self.name) + '_' + self.opt.log_str + 'iter%d' % self.iter + '.txt'
        self._z = np.loadtxt(join(self.opt.output_dir, 'segmentation', name))
        self._subact_count_update()


    ########### ADDED ##########################
    def find_segments(self, labels):
        """Finds contiguous segments in a list of labels."""
        segments = []
        for key, group in itertools.groupby(enumerate(labels), lambda ix : ix[1]):
            group = list(group)
            segments.append((key, group[0][0], group[-1][0]))  # (label, start_idx, end_idx)
        return segments


    def reorder_clusters(self, transcript, model):
        """Reorders clusters based on the transcript, which specifies the estimated order of actions"""

        # Find contiguous segments in the original self._z
        segments = self.find_segments(self._z)
        print("Original segments:", segments)

        # Normalize transcript labels to start from 0
        transcript = [(label - 1, start) for label, start in transcript]
        print("Normalized transcript:", transcript)

        # Create a list to hold the new segments based on the order given in the transcript
        new_segments = []
        
        for i, (transcript_label, transcript_start) in enumerate(transcript):
            # Find the segment in the original segments that corresponds to this transcript entry by position
            if i < len(segments):
                _, start, end = segments[i]
                # Append the segment with the new label to new_segments
                new_segments.append((transcript_label, start, end))

        # Sort the new segments based on their original start positions to maintain order
        new_segments.sort(key=lambda x: x[1])

        # Construct the new self._z based on the reordered segments
        new_z = np.full(self.n_frames, -1, dtype=int)
        for new_label, start, end in new_segments:
            print("Segment from {} to {} with new label {}".format(start, end, new_label))
            new_z[start:end + 1] = new_label

        self._z = new_z

        # Debug: Print the unique labels after reordering
        unique_labels = np.unique(self._z)
        print("Unique labels after reordering:", unique_labels)

        # Log changes in likelihood grid after reordering
        #logger.debug(f"Likelihood Grid after reordering for video {self.name}: Min {self._likelihood_grid.min()}, Max {self._likelihood_grid.max()}")

        return new_z

    def update_cluster_labels(self, new_labels):
        """Update the cluster labels with the new labels."""
        self._z = new_labels
        self._subact_count_update()
