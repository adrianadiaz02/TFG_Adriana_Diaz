#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

import sys
import os
sys.path.append(os.path.abspath('.').split('data_utils')[0])

from data_utils.BF_utils.update_argpars import update

from ute.utils.arg_pars import parser
from ute.ute_pipeline import temp_embed, all_actions
import json


if __name__ == '__main__':
    
    opt = parser.parse_args()
    opt.num_splits = 32

    opt.dataset_root = '/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/data' # set root
    
    
    if not os.path.exists(opt.exp_root):
        os.mkdir(opt.exp_root)

    opt.tensorboard_dir = os.path.join(opt.exp_root, opt.description)
    os.mkdir(opt.tensorboard_dir)   

    opt.learn_prototype = True
    # set activity
    # ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    # all
    # set feature extension and dimensionality
    opt.ext = 'txt'
    opt.feature_dim = 64

    # model name can be 'mlp' or 'nothing' for no embedding (just raw features)
    opt.model_name = 'mlp'

    # resume training
    opt.resume = False
    # load an already trained model (stored in the models directory in dataset_root)
    opt.load_model = False
    # opt.loaded_model_name = '%s.pth.tar'


    # USE TRANSFORMER!!!!
    opt.use_transformer = False
    """ scrambledegg: 7910       milk: 2524          sandwich: 3554      pancake: 9741       tea: 1661                                                                                                                                                                        
    friedegg: 8343           salat: 5693         coffee: 1208        juice: 2947         cereals: 1424      """
    if opt.use_transformer:
        opt.batch_size = 256
        opt.lr = 1e-6
        opt.weight_decay = 1e-4
        opt.transformer_num_layers = 2
        opt.transformer_dropout = 0.3
        opt.transformer_num_heads = 5
        opt.early_stopping = True

    # USE PERMUTATION AWARE PRIOR!!!
    opt.apply_permutation_aware_prior = True
    #opt.early_stopping = True


    # update log name and absolute paths
    opt = update(opt)

    # run temporal embedding
    if opt.subaction == 'all':
        #actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
        actions = ['cereals']
        
        if opt.apply_permutation_aware_prior:
            opt.transcripts_path = f'estimated_transcripts_{actions[0]}.json'
            #opt.transcripts_path = 'transcripts_GT_cereals.json'
            
            filename = opt.transcripts_path
            # Check if the file already exists to avoid overwriting existing data
            if not os.path.exists(filename):
                # Create an empty JSON file
                with open(filename, 'w') as file:
                    json.dump({}, file)  # Writes an empty dictionary as the JSON content
                print(f"Empty JSON file created at {filename}")
            
            else:
                print(f"File already exists: {filename}")

        all_actions(actions, opt)
    else:
        temp_embed(opt)

