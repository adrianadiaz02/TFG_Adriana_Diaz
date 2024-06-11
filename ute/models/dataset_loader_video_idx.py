#!/usr/bin/env python

"""Creating dataset out of video features for different models.
"""

__all__ = ''
__author__ = 'Anna Kukleva (base code), Adriana Díaz Soley (modifications)'
__date__ = 'December 2018, modified in May 2024'


from torch.utils.data import Dataset
import torch
import numpy as np
import random
from ute.utils.logging_setup import logger
from ute.utils.util_functions import join_data
import math

import json
import os

class FeatureDataset(Dataset):
    def __init__(self, videos, features, video_ids, video_names):
        logger.debug('Creating feature dataset')

        self._features = features
        self._gt = None
        # self._videos_features = features
        self._videos = videos

        ############ AÑADIDO ##########
        self._video_ids = video_ids  # Storing video identifiers
        self._video_names = video_names  # Storing video names
        ###############################

    def __len__(self):
        return len(self._gt)

    def __getitem__(self, idx):
        gt_item = self._gt[idx]
        features = self._features[idx]

        ############# AÑADIDO ##############
        video_id = self._video_ids[idx]  # Retrieve video ID
        video_name = self._video_names[idx]  # Retrieve video name
        ####################################
        return np.asarray(features), gt_item, video_id, video_name

    # @staticmethod
    # def _gt_transform(gt):
    #     if opt.model_name == 'tcn':
    #         return np.array(gt)[..., np.newaxis]
    #     if opt.model_name == 'mlp':
    #         return np.array(gt)

class VideoRelTimeDataset(FeatureDataset):
    def __init__(self, num_frames, num_splits, videos, features, video_ids, opt):
        logger.debug('Relative time labels')
        #super().__init__(videos, features)

        ######### AÑAIDDO ###########
        super().__init__(videos, features, video_ids)  # Pass video IDs to the superclass constructor
        #############################

        self._video_features_list = []  # used only if opt.concat > 1
        self._video_gt_list = []
        self._action_gt_list = []
        self.num_frames = num_frames
        self.num_splits = num_splits
        self.opt = opt
        

        print("Number of videos: {}".format(len(self._videos)))
        print("ITERNADO EN VIDEORELTIMEDATASET")
        for video in self._videos:
            print(video.name)
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            action_gt = video.gt
            #temp_features = join_data(temp_features, video_features, np.vstack)

            self._gt = join_data(self._gt, time_label, np.vstack)
            self._video_features_list.append(video_features)
            self._video_gt_list.append(time_label)
            self._action_gt_list.append(action_gt)

        print("Length of video dataset: {}".format(len(self._video_gt_list)))
        
    def __len__(self):
        return len(self._video_gt_list)

    def __getitem__(self, idx):
        
        gt_item = self._video_gt_list[idx]
        features = self._video_features_list[idx]
        gt_sample, features_sample  = self.uniform_sample(gt_item, features)

        ######### AÑADIDO ########
        video_id = self._video_ids[idx]  # Access video ID from the stored video IDs
        video_name = self._video_names[idx]  # Access video name from the stored video names
        ##########################
        
        return np.asarray(features_sample), gt_sample, video_id, video_name
    
    
    def random_sample(self, gt_item, features):
        
        sample_mask = np.sort(random.sample(list(np.arange(features.shape[0])), self.num_frames))
        return gt_item[sample_mask], features[sample_mask]

    def uniform_sample(self, gt_item, features):

        splits = np.arange(self.num_splits) *(math.floor(features.shape[0]/self.num_splits))
        splits = np.repeat(splits, self.num_frames/self.num_splits, axis = 0)
        indices = np.sort(splits + random.choices(list(np.arange(math.floor(features.shape[0]/self.num_splits))), k = self.num_frames))
  
        return gt_item[indices], features[indices]

class VideoRelTimeDataset_tcn(FeatureDataset):
    def __init__(self, num_frames, num_splits, videos, features, video_ids, video_names, opt):
        logger.debug('Relative time labels')
        #super().__init__(videos, features)
        
        ######### AÑADIDO ###########
        super().__init__(videos, features, video_ids, video_names)  # Pass video IDs to the superclass constructor
        #############################

        self._video_features_list = []  # used only if opt.concat > 1
        self._video_gt_list = []
        self._action_gt_list = []
        self.num_frames = num_frames
        self.num_splits = num_splits
        self.opt = opt
        

        print("Number of videos: {}".format(len(self._videos)))
        for video in self._videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            action_gt = video.gt
            #temp_features = join_data(temp_features, video_features, np.vstack)

            self._gt = join_data(self._gt, time_label, np.vstack)
            self._video_features_list.append(video_features)
            self._video_gt_list.append(time_label)
            self._action_gt_list.append(action_gt)
        print("Length of video dataset: {}".format(len(self._video_gt_list)))
        
    def __len__(self):
        return len(self._video_gt_list)

    def __getitem__(self, idx):
        
        gt_item = self._video_gt_list[idx]
        features = self._video_features_list[idx]
        gt_sample, features_sample  = self.uniform_sample(gt_item, features)

        ######### AÑADIDO ########
        video_id = self._video_ids[idx]  # Access video ID from the stored video IDs
        video_name = self._video_names[idx]  # Access video name from the stored video names
        ##########################
        
        return np.asarray(features_sample), gt_sample, video_id, video_name
    

    def random_sample(self, gt_item, features):
        
        sample_mask = np.sort(random.sample(list(np.arange(features.shape[0])), self.num_frames))
        return gt_item[sample_mask], features[sample_mask]

    def uniform_sample(self, gt_item, features):

        splits = np.arange(self.num_splits) *(math.floor(features.shape[0]/self.num_splits))
        splits = np.repeat(splits, (self.num_frames//2)/self.num_splits, axis = 0)
        indices = (splits + random.choices(list(np.arange(math.floor(features.shape[0]/self.num_splits))), k = (self.num_frames//2)))
        indices_positive = indices + np.array(random.choices(list(np.arange(-self.opt.window_size, self.opt.window_size)), k = indices.shape[0]))
        indices = np.sort(np.concatenate([indices, indices_positive]))
        indices = np.clip(indices, 0, features.shape[0] - 1)       
  
        return gt_item[indices], features[indices]





class GTDataset(FeatureDataset):
    def __init__(self, videos, features, video_ids, video_names):
        logger.debug('Ground Truth labels')
        super().__init__(videos, features, video_ids, video_names)  # Pass video IDs and names to the superclass constructor) - AÑADIDO

        for video in self._videos:
            gt_item = np.asarray(video.gt).reshape((-1, 1))
            # video_features = self._videos_features[video.global_range]
            # video_features = join_data(None, (gt_item, video_features),
            #                            np.hstack)
            self._gt = join_data(self._gt, gt_item, np.vstack)

            # self._features = join_data(self._features, video_features,
            #                            np.vstack)


class RelTimeDataset(FeatureDataset):
    def __init__(self, videos, features, video_ids, video_names):
        logger.debug('Relative time labels')
        super().__init__(videos, features, video_ids, video_names)  # Pass video IDs and names to the superclass constructor) - AÑADIDO

        temp_features = None  # used only if opt.concat > 1
        for video in self._videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            temp_features = join_data(temp_features, video_features, np.vstack)

            self._gt = join_data(self._gt, time_label, np.vstack)
            # video_features = join_data(None, (time_label, video_features),
            #                             np.hstack)

class TCNDataset(FeatureDataset):
    def __init__(self, videos, features, video_ids, video_names):

        logger.debug('Relative time labels')
        super().__init__(videos, features, video_ids, video_names)  # Pass video IDs and names to the superclass constructor) - AÑADIDO

        temp_features = None  # used only if opt.concat > 1
        for video in self._videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            
            
            pos_indices = np.arange(len(video_features)) + random.choices(range(1, 30), k = len(video_features))
            pos_indices = np.minimum(pos_indices, len(video_features) - 1)
            video_features_pos = video_features[pos_indices]
            video_features = np.concatenate([np.expand_dims(video_features, axis = 1), np.expand_dims(video_features_pos, axis = 1)], axis = 1)
            
        

            temp_features = join_data(temp_features, video_features, np.vstack)
            #print("Temp features: {}".format(temp_features.shape))

            self._gt = join_data(self._gt, time_label, np.vstack)
        self._features = temp_features

def load_ground_truth(videos, features, shuffle=True):
    logger.debug('load data with ground truth labels for training some embedding')

    dataset = GTDataset(videos, features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=opt.num_workers)
    return dataloader


def load_reltime_video_idx(videos, features, opt, mode="train", shuffle=True):
    logger.debug('load data with temporal labels as ground truth')
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    # Always generate video IDs!
    print("CREANDO IDENTIFICADORES DE LOS VIDEOS...")
    video_ids = [f"video_{i}" for i in range(len(videos))] 
    video_names = [video.name for video in videos]
    #print("number of videos: ", len(videos))
    #print("Video names: ", video_names)

    # Create a dictionary mapping video IDs to filenames
    video_id_to_name = {video_id: video_name for video_id, video_name in zip(video_ids, video_names)}

    # Save the dictionary as a JSON file
    mapping_file =  f'video_id_mapping_{opt.subaction}.json'
    mapping_file_path = os.path.join("video_id_mappings", mapping_file)
    with open(mapping_file_path, 'w') as mapping_file:
        json.dump(video_id_to_name, mapping_file)
    print(f"Video ID to name mapping saved to {mapping_file_path}")

    if opt.model_name == 'mlp':
        if mode == "train":
            print("Num frames: {}".format(opt.batch_size/opt.num_videos))

            ########## AÑADIDO #################
            dataset = VideoRelTimeDataset_tcn(num_frames = int(opt.batch_size/opt.num_videos), num_splits = opt.num_splits,
                                               videos = videos, features = features, video_ids = video_ids, video_names=video_names,
                                               opt = opt)
            print("shape dataset train: ", len(dataset))

            ###################################
            #dataset = VideoRelTimeDataset_tcn(num_frames = int(opt.batch_size/opt.num_videos), num_splits = opt.num_splits, videos = videos, features = features, opt = opt)
            
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.num_videos, 
                                                     shuffle=shuffle, num_workers=opt.num_workers)
        else:
            dataset = RelTimeDataset(videos, features, video_ids, video_names)

    if opt.model_name == 'tcn':
        dataset = TCNDataset(videos, features, video_ids, video_names)

    if mode == "test":
        print("EN TEEEST")
        print("shape dataset test: ", len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=opt.num_workers)

    return dataloader
