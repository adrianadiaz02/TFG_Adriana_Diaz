#!/usr/bin/env python

"""Module with Corpus class. There are methods for each step of the alg for the
whole video collection of one complex activity. See pipeline."""

__author__ = 'Anna Kukleva (base code), Adriana Díaz Soley (modifications)'
__date__ = 'August 2018, modified in May 2024'

import numpy as np
import os
import os.path as ops
import torch
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture
import time
from sklearn.cluster import MiniBatchKMeans

from ute.video import Video
from ute.models import mlp
from ute.utils.logging_setup import logger
from ute.eval_utils.accuracy_class import Accuracy
from ute.utils.mapping import GroundTruth
from ute.utils.util_functions import join_data, timing, dir_check, softmax, save_video_assignments
from ute.utils.visualization import Visual, plot_segm
from ute.probabilistic_utils.gmm_utils import AuxiliaryGMM, GMM_trh
from ute.eval_utils.f1_score import F1Score
from ute.models.dataset_loader import load_reltime
from ute.models.training_embed import load_model, training
import pickle

########### ADDED #####################
from ute.models.dataset_loader_video_idx import load_reltime_video_idx
import json

from ute.models import transformer
import torch.nn.functional as F

from ute.models.training_embed_permutation_aware import training_permutation_aware
#######################################

class Corpus(object):
    def __init__(self, opt, subaction='coffee', K=None):
        """
        Args:
            Q: number of Gaussian components in each mixture
            subaction: current name of complex activity
        """
        np.random.seed(opt.seed)
        self.gt_map = GroundTruth(frequency=opt.frame_frequency)
        self.gt_map.load_mapping(opt)
        self._K = self.gt_map.define_K(subaction=subaction) if K is None else K
        logger.debug('%s  subactions: %d' % (subaction, self._K))
        self.iter = 0
        self.return_stat = {}
        self._acc_old = 0
        self._videos = []
        self._subaction = subaction
        # init with ones for consistency with first measurement of MoF
        self._subact_counter = np.ones(self._K)
        self._gaussians = {}
        self._inv_count_stat = np.zeros(self._K)
        self._embedding = None
        self._gt2label = None
        self._label2gt = {}

        self._with_bg = opt.bg
        self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._features = None
        self._embedded_feat = None
        self.opt = opt
        self._init_videos()
        # logger.debug('min: %f  max: %f  avg: %f' %
        #              (np.min(self._features),
        #               np.max(self._features),
        #               np.mean(self._features)))

        # to save segmentation of the videos
        dir_check(os.path.join(opt.output_dir, 'segmentation'))
        dir_check(os.path.join(opt.output_dir, 'likelihood'))
        self.vis = None  # visualization tool

        ######### ADDED #############
        self.transcripts = None
        if opt.apply_permutation_aware_prior:
            self.transcripts = self.load_initial_transcripts(opt.transcripts_path)
        ###############################
        

    def _init_videos(self):
        #print("Init called")
        print("load emb value is: {}".format(self.opt.load_embed_feat))
        print(self.opt.data)
        logger.debug('.')
        gt_stat = Counter()

        for root, dirs, files in os.walk(self.opt.data):
            print(root)
            if not files:
                continue
            for filename in files:
                #print("HERE")
                # pick only videos with certain complex action
                # (ex: just concerning coffee)
                if self._subaction in filename:
                    if self.opt.test_set:
                        if self.opt.reduced:
                            self.opt.reduced = self.opt.reduced - 1
                            continue
                    # if self.opt.dataset == 'fs':
                    #     gt_name = filename[:-(len(self.opt.ext) + 1)] + '.txt'
                    # else:
                    match = re.match(r'(.*)\..*', filename)
                    gt_name = match.group(1)
                    # use extracted features from pretrained on gt embedding
                    if self.opt.load_embed_feat:
                        print("HEREE")
                        path = os.path.join(self.opt.data, 'embed', self.opt.subaction,
                                            self.opt.resume_str % self.opt.subaction) + '_%s' % gt_name
                    else:
                        path = os.path.join(root, filename)
                    
                    start = 0 if self._features is None else self._features.shape[0]
                    try:
                        video = Video(path, K=self._K,
                                      gt=self.gt_map.gt[gt_name],
                                      name=gt_name,
                                      start=start,
                                      with_bg=self._with_bg, opt = self.opt)
                    except AssertionError:
                        logger.debug('Assertion Error: %s' % gt_name)
                        continue
                    self._features = join_data(self._features, video.features(),
                                               np.vstack)

                    video.reset()  # to not store second time loaded features
                    self._videos.append(video)
                    # accumulate statistic for inverse counts vector for each video
                    gt_stat.update(self.gt_map.gt[gt_name])
                    if self.opt.reduced:
                        if len(self._videos) > self.opt.reduced:
                            break

                    if self.opt.feature_dim > 100:
                        if len(self._videos) % 20 == 0:
                            logger.debug('loaded %d videos' % len(self._videos))
        
        # update global range within the current collection for each video
        for video in self._videos:
            video.update_indexes(len(self._features))
        logger.debug('gt statistic: %d videos ' % len(self._videos) + str(gt_stat))
        self._update_fg_mask()

    def _update_fg_mask(self):
        logger.debug('.')
        if self._with_bg:
            self._total_fg_mask = np.zeros(len(self._features), dtype=bool)
            for video in self._videos:
                self._total_fg_mask[np.nonzero(video.global_range)[0][video.fg_mask]] = True
        else:
            self._total_fg_mask = np.ones(len(self._features), dtype=bool)

    def get_videos(self):
        for video in self._videos:
            yield video

    def get_features(self):
        return self._features

    def video_byidx(self, idx):
        return np.asarray(self._videos)[idx]

    def __len__(self):
        return len(self._videos)

    def regression_training(self):
        if self.opt.load_embed_feat:
            logger.debug('load precomputed features')
            self._embedded_feat = self._features
            return

        logger.debug('.')

        print("Learning Prototype: {}".format(self.opt.learn_prototype))

        dataloader = load_reltime_video_idx(videos=self._videos,
                                  features=self._features, opt = self.opt)
        
        if self.opt.use_transformer == False:
            print("creating mlp model...")
            model, loss, tcn_loss, optimizer = mlp.create_model(num_clusters = self._K, opt = self.opt ,  learn_prototype = self.opt.learn_prototype)
        
        else:
            #segment_length = self.opt.transformer_sequence_len
            #segmented_features = self.segment_features(self._features, segment_length)
            #print(segmented_features.size())
            print("creating transformer model...")
            model, loss, tcn_loss, optimizer = transformer.create_model(self.opt, self._K, self.opt.learn_prototype)

        
        if self.opt.load_model:
            model.load_state_dict(load_model(self.opt))
            self._embedding = model
        
        else:
            self._embedding = training(dataloader, self.opt.epochs,
                                       save=self.opt.save_model,
                                       model=model,
                                       loss=loss,
                                       tcn_loss = tcn_loss,
                                       learn_prototype = self.opt.learn_prototype,
                                       optimizer=optimizer,
                                       name=self.opt.model_name, opt = self.opt,
                                       corpus=self)  # Pass the current Corpus instance

        self._embedding = self._embedding.cpu()

        unshuffled_dataloader = load_reltime(videos=self._videos,
                                             features=self._features, 
                                             mode = "test", opt = self.opt,
                                             shuffle=False)


        gt_relative_time = None
        relative_time = None
        if self.opt.model_name == 'mlp':
            for batch_features, batch_gtreltime in unshuffled_dataloader:
                if self._embedded_feat is None:
                    self._embedded_feat = batch_features
                    print("After First Phase Training - Embedding Shape:", self._embedded_feat.shape)
                    if isinstance(self._embedded_feat, torch.Tensor):
                        print("After First Phase Training - Embedding Mean:", self._embedded_feat.mean().item())
                        logger.debug(f"After First Phase Training - Embedding Mean:: {self._embedded_feat.mean().item()}")
                    else:
                        print("After First Phase Training - Embedding Mean:", np.mean(self._embedded_feat))


                else:
                    self._embedded_feat = torch.cat((self._embedded_feat, batch_features), 0)

                batch_gtreltime = batch_gtreltime.numpy().reshape((-1, 1))
                gt_relative_time = join_data(gt_relative_time, batch_gtreltime, np.vstack)

            #relative_time = self._embedding(self._embedded_feat.float())[0].detach().numpy().reshape((-1, 1))
            

            self._embedded_feat = self._embedding.embedded(self._embedded_feat.float()).detach().numpy()
            self._embedded_feat = np.squeeze(self._embedded_feat)

        # if opt.save_embed_feat:
        #     self.save_embed_feat()

        # mse = np.sum((gt_relative_time - relative_time)**2)
        # mse = mse / len(relative_time)
        # logger.debug('MLP training: MSE: %f' % mse)

        return model

    ###################### ADDED ##############################
    def reset_for_second_training(self):
        # Reset all attributes except transcripts
        self._embedded_feat = None
        self._total_fg_mask = None
        self._features = None
        self._videos = []
        self._gaussians = {}
        self._subact_counter = np.ones(self._K)
        self.iter = 0

    def second_regression_training(self):
        """ Train a new model that uses the estimated transcripts from the previous regression training to 
        extract more accurate temporal embeddings that allow permutations between actions (for tha case of
        basic transcript) or/and repetition of actions (for the improved transcript)"""

        # Reset everything
        self.reset_for_second_training()

        self._init_videos()
        self._update_fg_mask()

        logger.debug('.')

        # Load estimated transcripts
        transcripts = self.transcripts
        logger.debug('load transcripts')
        logger.debug('transcripts:', transcripts)

        print("Learning Prototype: {}".format(self.opt.learn_prototype))

        dataloader = load_reltime_video_idx(videos=self._videos,
                                  features=self._features, opt = self.opt)
        
        if self.opt.use_transformer == False:
            print("creating mlp model...")
            model, loss, tcn_loss, optimizer = mlp.create_model(num_clusters = self._K, opt = self.opt ,  learn_prototype = self.opt.learn_prototype)
        
        else:
            print("creating transformer model...")
            model, loss, tcn_loss, optimizer = transformer.create_model(self.opt, self._K, self.opt.learn_prototype)

        
        if self.opt.load_model:
            model.load_state_dict(load_model(self.opt))
            self._embedding = model
        
        else:
            self._embedding = training_permutation_aware(dataloader, self.opt.epochs_second,
                                                         save=self.opt.save_model,
                                                         model=model,
                                                         loss=loss,
                                                         tcn_loss = tcn_loss,
                                                         learn_prototype = self.opt.learn_prototype,
                                                         optimizer=optimizer,
                                                         name=self.opt.model_name, opt = self.opt,
                                                         corpus=self,  # Pass the current Corpus instance
                                                         transcripts=transcripts)

        self._embedding = self._embedding.cpu()

        unshuffled_dataloader = load_reltime(videos=self._videos,
                                             features=self._features, 
                                             mode = "test", opt = self.opt,
                                             shuffle=False)


        gt_relative_time = None
        relative_time = None
        if self.opt.model_name == 'mlp':
            for batch_features, batch_gtreltime in unshuffled_dataloader:
                if self._embedded_feat is None:
                    self._embedded_feat = batch_features
                    print("After Second Phase Training - Embedding Shape:", self._embedded_feat.shape)
                    if isinstance(self._embedded_feat, torch.Tensor):
                        print("After Second Phase Training - Embedding Mean:", self._embedded_feat.mean().item())
                        logger.debug(f"After Second Phase Training - Embedding Mean:: {self._embedded_feat.mean().item()}")

                    else:
                        print("After Second Phase Training - Embedding Mean:", np.mean(self._embedded_feat))


                else:
                    self._embedded_feat = torch.cat((self._embedded_feat, batch_features), 0)

                batch_gtreltime = batch_gtreltime.numpy().reshape((-1, 1))
                gt_relative_time = join_data(gt_relative_time, batch_gtreltime, np.vstack)
            

            self._embedded_feat = self._embedding.embedded(self._embedded_feat.float()).detach().numpy()
            self._embedded_feat = np.squeeze(self._embedded_feat)

        return model


    #############################################################################

    def get_proto_likelihood(self, features, prototypes):
        scores =  np.matmul(features, np.transpose(prototypes))
        #logger.debug(f"Scores Shape: {scores.shape}")
        #logger.debug(f"Scores Range: Min {scores.min()}, Max {scores.max()}")

        # Scale scores to avoid numerical instability
        scores = scores - np.max(scores, axis=1, keepdims=True)

        probs = softmax(scores/0.1)
        probs = np.clip(probs, 1e-30, 1)

        #logger.debug(f"Probabilities Shape: {probs.shape}")
        #logger.debug(f"Probabilities Range: Min {probs.min()}, Max {probs.max()}")

        #print(probs[0])
        return np.log(probs)
    
    def generate_prototype_likelihood(self, model):
        prototypes = model.get_prototypes().cpu().numpy()

        # Debugging prototypes
        #logger.debug(f"Prototypes Shape: {prototypes.shape}")
        #logger.debug(f"Prototypes: {prototypes}")

        for video_idx in range(len(self._videos)):
            self._video_likelihood_grid_proto(video_idx, prototypes)

        if self.opt.bg:
            scores = None
            for video in self._videos:
                scores = join_data(scores, video.get_likelihood(), np.vstack)
            
            bg_trh_score = np.sort(scores, axis=0)[int((self.opt.bg_trh / 100) * scores.shape[0])]
            trh_set = []
            for action_idx in range(self._K):
                trh_set.append(bg_trh_score[action_idx])
            
            for video in self._videos:
                video.valid_likelihood_update(trh_set)
    
        # Log likelihood grid after generation
        #logger.debug('Likelihood grid after generating prototype likelihoods:')
        #for video in self._videos:
        #    logger.debug(f"Video {video.name}: Likelihood Grid Range: Min {video._likelihood_grid.min()}, Max {video._likelihood_grid.max()}")  
    
    def _video_likelihood_grid_proto(self, video_idx, prototypes):
        video = self._videos[video_idx]
        if self.opt.load_embed_feat:
            features = self._features[video.global_range]
        else:
            features = self._embedded_feat[video.global_range]
        
        # Debugging features
        #logger.debug(f"Video {video.name}: Features Shape: {features.shape}")
        #logger.debug(f"Video {video.name}: Features Range: Min {features.min()}, Max {features.max()}")
    
        score_matrix = self.get_proto_likelihood(features, prototypes)

        # Debugging score matrix
        #logger.debug(f"Video {video.name}: Score Matrix Shape: {score_matrix.shape}")
        #logger.debug(f"Video {video.name}: Score Matrix Range: Min {score_matrix.min()}, Max {score_matrix.max()}")

        for subact in range(self._K):
            scores = score_matrix[:, subact]
            # Debugging scores for each subact
            #logger.debug(f"Video {video.name}: Subact {subact}: Scores Range: Min {scores.min()}, Max {scores.max()}")      

            video.likelihood_update(subact, scores)
            
    @timing
    def _gaussians_fit(self):
        """ Fit GMM to video features.

        Define subset of video collection and fit on it gaussian mixture model.
        If idx_exclude = -1, then whole video collection is used for comprising
        the subset, otherwise video collection excluded video with this index.
        Args:
            idx_exclude: video to exclude (-1 or int in range(0, #_of_videos))
            save: in case of mp lib all computed likelihoods are saved on disk
                before the next step
        """
        for k in range(self._K):
            gmm = GaussianMixture(n_components=1,
                                  covariance_type='full',
                                  max_iter=150,
                                  random_state=self.opt.seed,
                                  reg_covar=1e-4)
            total_indexes = np.zeros(len(self._features), dtype=np.bool)
            for idx, video in enumerate(self._videos):
                indexes = np.where(np.asarray(video._z) == k)[0]
                if len(indexes) == 0:
                    continue
                temp = np.zeros(video.n_frames, dtype=np.bool)
                temp[indexes] = True
                total_indexes[video.global_range] = temp

            total_indexes = np.array(total_indexes, dtype=np.bool)
            if self.opt.load_embed_feat:
                feature = self._features[total_indexes, :]
            else:
                feature = self._embedded_feat[total_indexes, :]
            time1 = time.time()
            try:
                gmm.fit(feature)
            except ValueError:
                gmm = AuxiliaryGMM()
            time2 = time.time()
            # logger.debug('fit gmm %0.6f %d ' % ((time2 - time1), len(feature))
            #              + str(gmm.converged_))

            self._gaussians[k] = gmm

        if self.opt.bg:
            # with bg model I assume that I use only one component
            for gm_idx, gmm in self._gaussians.items():
                self._gaussians[gm_idx] = GMM_trh(gmm)

    @timing
    def gaussian_model(self):
        logger.debug('Fit Gaussian Mixture Model to the whole dataset at once')
        self._gaussians_fit()
        for video_idx in range(len(self._videos)):
            self._video_likelihood_grid(video_idx)

        if self.opt.bg:
            scores = None
            for video in self._videos:
                scores = join_data(scores, video.get_likelihood(), np.vstack)

            with open("test_gmm.npy", "wb") as f:
                np.save(f, np.sort(scores, axis=0))
                bg_trh_score = np.sort(scores, axis=0)[int((self.opt.bg_trh / 100) * scores.shape[0])]
                np.save(f, bg_trh_score)
                bg_trh_set = []
                for action_idx in range(self._K):
                    
                    new_bg_trh = self._gaussians[action_idx].mean_score - bg_trh_score[action_idx]
                    if action_idx == 0:
                        np.save(f, self._gaussians[action_idx].mean_score)
                        np.save(f, new_bg_trh)

                    self._gaussians[action_idx].update_trh(new_bg_trh=new_bg_trh, opt=self.opt)
                    bg_trh_set.append(new_bg_trh)

                logger.debug('new bg_trh: %s' % str(bg_trh_set))
                trh_set = []
                for action_idx in range(self._K):
                    trh_set.append(self._gaussians[action_idx].trh)
                np.save(f, trh_set)
                for video in self._videos:
                    video.valid_likelihood_update(trh_set)

    def _video_likelihood_grid(self, video_idx):
        video = self._videos[video_idx]
        if self.opt.load_embed_feat:
            features = self._features[video.global_range]
        else:
            features = self._embedded_feat[video.global_range]
        for subact in range(self._K):
            scores = self._gaussians[subact].score_samples(features)
            if self.opt.bg:
                video.likelihood_update(subact, scores,
                                        trh=self._gaussians[subact].trh)
            else:
                video.likelihood_update(subact, scores)
        if self.opt.save_likelihood:
            video.save_likelihood()

    def get_proto_labels(self, embeddings, prototypes):
        print("Embeddings shape:", embeddings.shape)
        print("Prototypes shape:", prototypes.shape)

        prob_scores = np.matmul(embeddings, np.transpose(prototypes))
        assignments = np.argmax(prob_scores, axis = 1)
        assignments = (assignments)
        return assignments
    

    def compute_euclidean_dist_numpy(self, embeddings, prototypes):
        dists = -2 * np.dot(embeddings, prototypes.T) + np.sum(prototypes**2, axis = 1) +  np.sum(embeddings**2, axis = 1)[:, np.newaxis]
        
        #print(dists.shape)
        return dists

    def cluster_prototype(self, model):

        accuracy = Accuracy(self.opt)
        long_gt = []
        long_rt = []
        for video in self._videos:
            long_gt += list(video.gt)
            long_rt += list(video.temp)
        long_rt = np.array(long_rt)

        
        logger.debug("Performing Prototype based clustering")
        print("Shape of embeddings sent to Kmeans: {}".format(self._embedded_feat[self._total_fg_mask].shape))
        cluster_labels_orig = self.get_proto_labels(self._embedded_feat[self._total_fg_mask], model.get_prototypes().cpu().numpy())
        cluster_labels = cluster_labels_orig.copy()

        print("unique clusters: {}".format(np.unique(cluster_labels)))
        time2label = {}
        count_array = np.array(np.unique(cluster_labels_orig, return_counts=True)).T
        print("Counts of individual clusters: {}".format(count_array))
        
        for label in np.unique(cluster_labels):  
            cluster_mask = cluster_labels == label
            r_time = np.mean(long_rt[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        logger.debug('time ordering of labels')
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            cluster_labels[cluster_labels_orig == label] = time_idx

        shuffle_labels = np.arange(len(time2label))

        labels_with_bg = np.ones(len(self._total_fg_mask)) * -1

        # use predefined by time order  for kmeans clustering
        labels_with_bg[self._total_fg_mask] = cluster_labels # Apply initial clustering order
        print("labels_with_bg: ", len(labels_with_bg)) # >> 129551
        print("cluster_labels: ", len(cluster_labels)) # >> 129551

        logger.debug('Order of labels: %s %s' % (str(shuffle_labels), str(sorted(time2label))))
        accuracy.predicted_labels = labels_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass

        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        print(f"MoF Value before reordering: {str(accuracy.mof_val())}")

        #################################

        ########################################################################
        # VISUALISATION
        if self.opt.vis and self.opt.vis_mode != 'segm':
            dot_path = ''
        
            self.vis = Visual(mode=self.opt.vis_mode, opt = self.opt, save=True, svg=False, saved_dots=dot_path)
            self.vis.fit(self._embedded_feat[self._total_fg_mask], np.array(long_gt)[self._total_fg_mask], 'gt_', reset=False)
            self.vis.color(np.array(long_rt)[self._total_fg_mask], 'time_')
            self.vis.color(cluster_labels_orig, 'kmean')
        
        ########################################################################
        logger.debug('Update video z for videos before GMM fitting')
        labels_with_bg[labels_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(labels_with_bg[video.global_range])
            video.segmentation['cl'] = (video._z, self._label2gt)
            #save_video_assignments(video.name, video.segmentation, "Before Viterbi", self.opt.tensorboard_dir)

    
    def clustering(self):
        logger.debug('.')
        np.random.seed(self.opt.seed)

        kmean = MiniBatchKMeans(n_clusters=self._K,
                                 random_state=self.opt.seed,
                                 batch_size=50)
        print(self._embedded_feat.shape)
        kmean.fit(self._embedded_feat[self._total_fg_mask])
        

        accuracy = Accuracy(self.opt)
        long_gt = []
        long_rt = []
        for video in self._videos:
            long_gt += list(video.gt)
            long_rt += list(video.temp)
        long_rt = np.array(long_rt)

        kmeans_labels = np.asarray(kmean.labels_).copy()
        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(long_rt[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        logger.debug('time ordering of labels')
        print("time ordering of labels")
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            kmeans_labels[kmean.labels_ == label] = time_idx

        shuffle_labels = np.arange(len(time2label))

        labels_with_bg = np.ones(len(self._total_fg_mask)) * -1

        # use predefined by time order  for kmeans clustering
        labels_with_bg[self._total_fg_mask] = kmeans_labels

        logger.debug('Order of labels: %s %s' % (str(shuffle_labels), str(sorted(time2label))))
        accuracy.predicted_labels = labels_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass

        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        ########################################################################
        # VISUALISATION
        if self.opt.vis and self.opt.vis_mode != 'segm':
            dot_path = ''
            self.vis = Visual(mode=self.opt.vis_mode, opt = self.opt, save=True, svg=False, saved_dots=dot_path)
            self.vis.fit(self._embedded_feat[self._total_fg_mask], np.array(long_gt)[self._total_fg_mask], 'gt_', reset=False)
            self.vis.color(np.array(long_rt)[self._total_fg_mask], 'time_')
            self.vis.color(kmean.labels_, 'kmean')
        ########################################################################

        logger.debug('Update video z for videos before GMM fitting')
        labels_with_bg[labels_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(labels_with_bg[video.global_range])

        for video in self._videos:
            video.segmentation['cl'] = (video._z, self._label2gt)

    ########### ADDED ################################
    def load_initial_transcripts(self, path):
        """Load the initial transcripts given their corresponding path"""
        
        print(f"Loading initial transcripts from {path}")
        with open(path, 'r') as file:
            transcripts = json.load(file)
        return transcripts


    def update_transcripts(self, new_transcripts):
        """ Update the transcripts of the class"""
        
        print("Updating transcripts...")
        self.transcripts = new_transcripts
        print("Transcripts updated")


    def apply_transcript_reordering(self, model):
        """Apply transcript data to reorder clusters for each video. It relabels the segments based
        on the order of actions estimated by the transcript"""
        
        logger.debug("Applying transcript reordering")
        print("Applying transcript reordering")
        
        if self.opt.recluster_after_reordering:
            embeddings = []
            initial_labels = []
        
        for video_name, transcript in self.transcripts.items():
            print(video_name, transcript)
            
            video = next((v for v in self._videos if v.name == video_name), None)
            if video is None:
                print(f"Video {video_name} not found")
                continue
            
            # Relabel the segments based on the order of actions defined by the estimated transcript
            ordered_labels = video.reorder_clusters(transcript, model)
            print(f"Updated _z for {video_name}") 

            # Update the video labels
            video.update_cluster_labels(ordered_labels)
            
            # Save all updated embeddings and initial labels from each video
            if self.opt.recluster_after_reordering:
                embeddings.append(video.features())
                initial_labels.append(ordered_labels)
        
        
        if self.opt.recluster_after_reordering:
            # Concatenate all updated embeddings and initial labels from each video
            embeddings = np.concatenate(embeddings, axis=0)
            initial_labels = np.concatenate(initial_labels, axis=0)
            print(f"Initial labels after reordering: {np.unique(initial_labels)}")

            # Perform clustering again with the embeddings and initial labels
            #self.recompute_centroids_and_reassign(embeddings, initial_labels, model)
            self.recompute_centroids_and_cluster(embeddings, initial_labels, model)
            print("Reclustering with updated centroids done.")

    
    def recompute_centroids_and_reassign(self, embeddings, initial_labels, model):
        """Recompute centroids based on the initial classification and reassign embeddings to the closest prototype."""
        logger.debug("Recomputing centroids and reassigning embeddings")

        # Convert embeddings to tensor and get device
        device = next(model.parameters()).device
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        
        # Transform embeddings to the embedding space
        embeddings_transformed = model.embedded(embeddings_tensor).detach().cpu().numpy()

        # Initialize centroids
        centroids = []
        for label in range(self._K):
            centroid = embeddings_transformed[initial_labels == label].mean(axis=0)
            centroids.append(centroid)
        
        # Stack centroids to form prototype tensor
        centroids = torch.tensor(centroids, dtype=torch.float32).to(device)

        # Check if dimensions match
        if centroids.shape[1] != model.prototype_layer.weight.shape[1]:
            logger.debug(f"Centroids shape: {centroids.shape}")
            logger.debug(f"Model prototype layer weight shape: {model.prototype_layer.weight.shape}")
            raise ValueError(f"Dimension mismatch: Centroids have {centroids.shape[1]} features, but model prototypes have {model.prototype_layer.weight.shape[1]} features.")


        model.update_prototypes(centroids.cpu().numpy())
        
        # Reassign each embedding to the closest prototype
        embeddings_transformed_tensor = torch.tensor(embeddings_transformed, dtype=torch.float32).to(device)
        distances = torch.cdist(embeddings_transformed_tensor, centroids)
        assigned_labels = torch.argmin(distances, dim=1).cpu().numpy()
        
        # Update labels for each video
        start_idx = 0
        for video in self._videos:
            end_idx = start_idx + video.n_frames
            video.update_cluster_labels(assigned_labels[start_idx:end_idx])
            start_idx = end_idx

        logger.debug("Reassignment complete")


    def recompute_centroids_and_cluster(self, embeddings, initial_labels, model):
        """Recompute centroids based on the initial classification and perform 
        clustering with the given embeddings."""
        
        logger.debug("Recomputing centroids and performing new clustering")

        device = next(model.parameters()).device

        # Transform embeddings to the embedding space
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        embeddings = model.embedded(embeddings).detach().cpu().numpy()

        # Initialize the k-means algorithm
        kmean = MiniBatchKMeans(n_clusters=self._K, random_state=self.opt.seed, batch_size=50, max_iter=1)
        
        # Use the initial labels as a starting point for clustering
        unique_labels = np.unique(initial_labels)
        
        # For each unique label, calculate the mean of the embeddings that belong to that label.
        initial_centroids = np.array([embeddings[initial_labels == label].mean(axis=0) for label in unique_labels])
        logger.debug(f"Initial centroids shape: {initial_centroids.shape}")
        logger.debug(f"Initial Centroids: {initial_centroids}")

        # Perform KMeans clustering with initial centroids
        kmean.cluster_centers_ = initial_centroids
        kmean.fit(embeddings)
        
        # Get the cluster labels
        cluster_labels = np.asarray(kmean.labels_).copy()
        logger.debug(f"KMeans labels: {np.unique(cluster_labels)}")

        # Update the labels for each video
        start_idx = 0
        for video in self._videos:
            end_idx = start_idx + video.n_frames
            video.update_cluster_labels(cluster_labels[start_idx:end_idx])
            start_idx = end_idx

        # Update the model prototypes (centroids) based on the new clustering
        model.update_prototypes(kmean.cluster_centers_)
        
        updated_prototypes = model.get_prototypes()
        logger.debug(f"Updated prototypes: {updated_prototypes}")

    ##############################################################

    def _count_subact(self):
        self._subact_counter = np.zeros(self._K)
        for video in self._videos:
            self._subact_counter += video.a

    @timing
    def viterbi_decoding(self):
        logger.debug('.')
        self._count_subact()
        pr_orders = []
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            if self.opt.bg:
                video.update_fg_mask()

            # Debugging statements
            #logger.debug(f"Video {video.name}:")
            #logger.debug(f"  Likelihood Grid Shape: {video._likelihood_grid.shape}")
            #logger.debug(f"  Likelihood Grid Range: Min {video._likelihood_grid.min()}, Max {video._likelihood_grid.max()}")
            #logger.debug(f"  Likelihood Grid Type: {video._likelihood_grid.dtype}")
            #logger.debug(f"  Pi: {video._pi}")
            #print(f"  Foreground Mask: {video.fg_mask}")

            video.viterbi()
            cur_order = list(video._pi)
            if cur_order not in pr_orders:
                logger.debug(str(cur_order))
                pr_orders.append(cur_order)
        self._count_subact()
        logger.debug(str(self._subact_counter))


    def without_temp_emed(self):
        print("HEREEEEEE")
        logger.debug('No temporal embedding')
        self._embedded_feat = self._features.copy()

    @timing
    def accuracy_corpus(self, prefix=''):
        """Calculate metrics as well with previous correspondences between
        gt labels and output labels"""

        #print("Called with Prefix: {}".format(prefix))
        accuracy = Accuracy(self.opt)
        f1_score = F1Score(K=self._K, n_videos=len(self._videos))
        long_gt = []
        long_pr = []
        long_rel_time = []
        self.return_stat = {}

        for video in self._videos:
            long_gt += list(video.gt)
            long_pr += list(video._z)
            try:
                long_rel_time += list(video.temp)
            except AttributeError:
                pass
                # logger.debug('no poses')
        accuracy.gt_labels = long_gt
        accuracy.predicted_labels = long_pr
        if self.opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]

        old_mof, total_fr = accuracy.mof(old_gt2label=self._gt2label)
        self._gt2label = accuracy._gt2cluster
        self._label2gt = {}
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass
        acc_cur = accuracy.mof_val()
        logger.debug('%sAction: %s' % (prefix, self._subaction))
        logger.debug('%sMoF val: ' % prefix + str(acc_cur))

        logger.debug('%sprevious dic -> MoF val: ' % prefix + str(float(old_mof) / total_fr))

        accuracy.mof_classes()
        accuracy.iou_classes()

        self.return_stat = accuracy.stat()

        f1_score.set_gt(long_gt)
        f1_score.set_pr(long_pr)
        f1_score.set_gt2pr(self._gt2label)
        if self.opt.bg:
            f1_score.set_exclude(-1)
        f1_score.f1()

        for key, val in f1_score.stat().items():
            self.return_stat[key] = val
        
        for key, val in self.return_stat.items():
            print(key, val)



        for video in self._videos:
            video.segmentation[video.iter] = (video._z, self._label2gt)
            if prefix == "final":
                save_video_assignments(video.name, video.segmentation, "After Viterbi", self.opt.tensorboard_dir)
            else:
                save_video_assignments(video.name, video.segmentation, "Before Viterbi", self.opt.tensorboard_dir)
        if self.opt.vis:
            ########################################################################
            # VISUALISATION

            if self.opt.vis_mode != 'segm':
                long_pr = [self._label2gt[i] for i in long_pr]

                if self.vis is None:
                    self.vis = Visual(mode=self.opt.vis_mode, opt = self.opt, save=True, reduce=None)
                    self.vis.fit(self._embedded_feat, long_pr, 'iter_%d' % self.iter)
                else:
                    reset = prefix == 'final'
                    self.vis.color(labels=np.array(long_pr)[self._total_fg_mask], prefix='iter_%d' % self.iter, reset=reset)
            else:
                ####################################################################
                # visualisation of segmentation
                if prefix == 'final':
                    colors = {}
                    cmap = plt.get_cmap('tab20')
                    for label_idx, label in enumerate(np.unique(long_gt)):
                        if label == -1:
                            colors[label] = (0, 0, 0)
                        else:
                            # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
                            colors[label] = cmap(label_idx / len(np.unique(long_gt)))

                    dir_check(os.path.join(self.opt.tensorboard_dir, 'plots'))
                    dir_check(os.path.join(self.opt.tensorboard_dir, 'plots', self.opt.subaction))
                    fold_path = os.path.join(self.opt.tensorboard_dir, 'plots', self.opt.subaction, 'segmentation')
                    dir_check(fold_path)
                    ############
                    """# Example of checking data before plotting
                    for video in self._videos[:5]:  # Limit to first 5 for brevity
                        print(f"Video: {video.name}, Segments: {video.segmentation}")"""
                    ############
                    for video in self._videos:
                        path = os.path.join(fold_path, video.name + '.png')
                        name = video.name.split('_')
                        name = '_'.join(name[-2:])
                        plot_segm(path, video.segmentation, colors, name=name)
                ####################################################################
            ####################################################################

        return accuracy.frames()

    def resume_segmentation(self):
        logger.debug('resume precomputed segmentation')
        for video in self._videos:
            video.iter = self.iter
            video.resume()
        self._count_subact()

    def save_embed_feat(self):
        dir_check(ops.join(self.opt.data, 'embed'))
        dir_check(ops.join(self.opt.data, 'embed', self.opt.subaction))
        for video in self._videos:
            video_features = self._embedded_feat[video.global_range]
            feat_name = self.opt.resume_str + '_%s' % video.name
            np.savetxt(ops.join(self.opt.data, 'embed', self.opt.subaction, feat_name), video_features)