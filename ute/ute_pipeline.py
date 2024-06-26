"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva (base code), Adriana Díaz Soley (modifications)'
__date__ = 'June 2019, modified in May 2024'

from ute.corpus import Corpus
from ute.utils.logging_setup import logger
from ute.utils.util_functions import timing, update_opt_str, join_return_stat, parse_return_stat
import os

@timing
def temp_embed(opt):
    
    corpus = Corpus(subaction=opt.subaction, opt = opt) # loads all videos, features, and gt

    logger.debug('Corpus with poses created')
    print(opt.model_name)
    
    if opt.model_name in ['mlp']:
        # train or load a new model and use it to extract temporal embeddings for each video
        model = corpus.regression_training()
    
    if opt.model_name == 'nothing':
        print("HEREE in Nothing")
        corpus.without_temp_emed()

    # train a new model that uses the estimated transcripts from the previous regression training to 
    # extract more accurate temporal embeddings (that allow permutation or repetition of actions)
    #if opt.apply_permutation_aware_prior:
    if opt.do_2nd_train:
        model = corpus.second_regression_training()

    # Cluster the embeddings
    if not opt.learn_prototype: 
        print("HEREEEE")
        corpus.clustering()
    else:
        corpus.cluster_prototype(model)
        #corpus.clustering()
    
    if not opt.learn_prototype:
        corpus.gaussian_model()
    else:
        #corpus.gaussian_model()
        corpus.generate_prototype_likelihood(model)
    
    corpus.accuracy_corpus()

    if opt.resume_segmentation:
        corpus.resume_segmentation()
    else:
        corpus.viterbi_decoding()

   
    # Apply transcript reordering after Viterbi decoding
    if opt.apply_permutation_aware_prior:
        corpus.apply_transcript_reordering(model) # Also reclusters if True

        # Recompute the likelihoods after transcript reordering
        if opt.recluster_after_reordering:
            if opt.learn_prototype:
                corpus.generate_prototype_likelihood(model)
            else:
                corpus.gaussian_model()

            corpus.viterbi_decoding()

    corpus.accuracy_corpus('final')

    return corpus.return_stat


@timing
def all_actions(actions, opt):
    return_stat_all = None
    lr_init = opt.lr
    #freeze_iters_list = opt.freeze_iters
    for i, action in enumerate(actions):
        opt.subaction = action 
        #opt.freeze_iters = freeze_iters_list[i] 
        if not opt.resume:
            opt.lr = lr_init
        update_opt_str(opt)
        
        return_stat_single = temp_embed(opt)

        
        print("WRITTING OUTPUT FILES (FINAL_RESULTS)")
        print(opt.description)

        out_file = open(os.path.join(opt.dataset_root, opt.tensorboard_dir, "final_results_{}.txt".format(action)), "w")
        out_file_2 = open(os.path.join(opt.dataset_root, opt.tensorboard_dir, "final_results_{}_combined.txt".format(action)), "w")
        parse_return_stat(return_stat_single, out_file)    
        return_stat_all = join_return_stat(return_stat_all, return_stat_single)
        parse_return_stat(return_stat_all, out_file_2)
        out_file.close()
        out_file_2.close()

    logger.debug(return_stat_all)
    out_file = open(os.path.join(opt.dataset_root, opt.tensorboard_dir, "final_results.txt"), "w")
    #out_file = open(os.path.join(opt.dataset_root, "test_runs", "final_results.txt"), "w")
    
    parse_return_stat(return_stat_all, out_file)


def resume_segmentation(iterations=10, opt=any):
    logger.debug('Resume segmentation')
    corpus = Corpus(subaction=opt.action)

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.resume_segmentation()
        corpus.accuracy_corpus()
    corpus.accuracy_corpus()