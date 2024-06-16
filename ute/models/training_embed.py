#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model']
__author__ = 'Anna Kukleva (base code), Adriana Díaz Soley (modifications)'
__date__ = 'August 2018, modified in May 2024'

import torch
import torch.backends.cudnn as cudnn
from os.path import join
import time
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import io
from imageio import imread
from ute.utils.logging_setup import logger
from ute.utils.util_functions import Averaging, adjust_lr
from ute.utils.util_functions import dir_check, save_params
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import json
from scipy.signal import find_peaks
import os
from codecarbon import EmissionsTracker

#############ADDED ######################
class EarlyStopping:
    """Early stop the training if the loss doesn't improve or changes minimally after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait after last time the loss changed significantly.
            verbose (bool): If True, prints a message for each significant change in loss.
            delta (float): Minimum change in the loss to qualify as a significant change.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, current_loss):
        if self.best_score is None:
            self.best_score = current_loss
            return False

        if abs(self.best_score - current_loss) > self.delta:
            self.best_score = current_loss
            self.counter = 0
            if self.verbose:
                print(f'Significant change detected. Resetting counter. Best loss: {self.best_score}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'No significant change. Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
def save_transcripts(opt, estimated_transcripts_all, corpus):
    """Save the transcript in a json file in the defined tensorboard directory."""

    tensorboard_dir = '/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs'
    filename = f'estimated_transcripts_{opt.subaction}.json'
    path = os.path.join(tensorboard_dir, opt.description, filename)
    
    try:
        # Save the estimated transcripts for each video
        print("GUARDANDO TRANSCRIPT")
        with open(path, 'w') as file:
            json.dump(estimated_transcripts_all, file)
        logger.info(f"Successfully saved estimated transcripts to {path}.")
        print(f"Successfully saved estimated transcripts to {path}.")
    
    except IOError as e:
        logger.error(f"Failed to save estimated transcripts: {str(e)}")
    
    corpus.update_transcripts(estimated_transcripts_all)


def prepare_transcripts(opt, estimated_transcripts, estimated_transcripts_all, video_ids):
    """ Prepares the transcript to be saved by assigning the video identefier to its 
    corresponding name. """
    # Get the dictionary that maps video_ids to their names
    mapping_path = os.path.join("video_id_mappings", f'video_id_mapping_{opt.subaction}.json')

    with open(mapping_path, 'r') as file:
        video_name_mapping = json.load(file)

    for video_id in video_ids:
        if video_id in video_name_mapping:
            # instead of saving with the video_id, save it with its corresponding name
            descriptive_name = video_name_mapping[video_id]  # Get the corresponding descriptive name
            estimated_transcripts_all[descriptive_name] = estimated_transcripts
        else:
            logger.error(f"Video ID {video_id} not found in video_name_mapping")
    return estimated_transcripts_all

#############################################################


def training(train_loader, epochs, save, corpus, **kwargs):
    """Training pipeline for embedding.

    Args:
        train_loader: iterator within dataset
        epochs: how much training epochs to perform
        n_subact: number of subactions in current complex activity
        mnist: if training with mnist dataset (just to test everything how well
            it works)
    Returns:
        trained pytorch model
    """
    
    logger.debug('create model')


    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    learn_prototype = kwargs['learn_prototype']
    tcn_loss = kwargs['tcn_loss']
    opt = kwargs['opt']

    if opt.early_stopping:
        print("patience: ", opt.early_stop_patience)
        print("delta: ", opt.early_stop_delta)
        early_stopper = EarlyStopping(patience=opt.early_stop_patience, verbose=True, delta=opt.early_stop_delta)
        early_stopped = False
    
    # make everything deterministic -> seed setup
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(opt.tensorboard_dir)
    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()
    c_losses = Averaging()
    tcn_losses = Averaging()

    adjustable_lr = opt.lr

    #################### ADDED ########################
    # Files that track the 3 training losses (cluster, tcn, and final losses)
    TCN_losses_path = join(opt.tensorboard_dir, 'TCN_Losses.txt')
    cluster_losses_path = join(opt.tensorboard_dir, 'Cluster_Losses.txt')
    final_losses_path = join(opt.tensorboard_dir, 'Final_Losses.txt')
    ###################################################


    logger.debug('epochs: %s', epochs)
    f = open("test_q_distribution.npy", "wb")

    #################### ADDED ########################
    torch.autograd.set_detect_anomaly(True)

    if opt.apply_permutation_aware_prior:
        estimated_transcripts_all = {}

    
    # Start emissions tracking
    tensorboard_dir = '/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs'
    path = os.path.join(tensorboard_dir, opt.description, 'regression_training')

    # Create the directory
    if not os.path.exists(path):
        os.makedirs(path)

    tracker = EmissionsTracker(output_dir=path, project_name="regression_training")
    tracker.start()

    #########################################

    for epoch in range(epochs):
        # model.cuda()
        model.to(opt.device)
        model.train()

        logger.debug('Epoch # %d' % epoch)
      
        end = time.time()

        total_loss = 0 # For early stopping
      
        for i, (features, labels, video_ids, video_names) in enumerate(train_loader): # videos_ids y video_names ADDED

            #print(f"Processing {video_ids}")

            num_videos = features.shape[0] 
            #print("Number of videos:", num_videos) # >> 2 o 1(Number of videos in a batch)
            
            if opt.use_transformer:
                # Add check for NaNs in features
                if not torch.isfinite(features).all():
                    raise ValueError(f"Features contain NaN or Inf at batch {i}, video_ids {video_ids}") # obs: Features don't contain NaN or Inf
                # Add check for NaNs in labels
                if not torch.isfinite(labels).all():
                    raise ValueError(f"Labels contain NaN or Inf at batch {i}, video_ids {video_ids}") # obs: Labels don't contain NaN or Inf

                features = features.transpose(0, 1)  # Transpose to [sequence_length, batch_size, feature_dim]
                labels =  labels.transpose(0,1)

            if not opt.use_transformer:
                #print("Features shape: ", features.shape) # >> [2, 128, 64] (SIEMPRE), 2 videos, batch size 256
                features = torch.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
                #print("Features shape after reshape: ", features.shape) # >> [256, 64] 
                            
                labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1], labels.shape[2]))
            

            data_time.update(time.time() - end)
            features = features.float()
            labels = labels.float().to(opt.device)


            if opt.device == 'cuda':
                features = features.cuda(non_blocking=True)

            output, proto_scores, embs = model(features)
            
            #print("output: ", output.shape) # >> [512, 1]
            #print("proto scores: ", proto_scores.shape) # >> [512, 7]
            #print("embs: ", embs.shape) # >> [512, 40]


            if learn_prototype:
                with torch.no_grad():

                    #compute q
                    p_gauss = get_cost_matrix(batch_size = opt.batch_size, num_videos = num_videos, num_videos_dataset = opt.num_videos \
                    ,sigma = opt.sigma, num_clusters = proto_scores.shape[1]) # Prior distribution, maintains fixed order of actions

                    if opt.apply_temporal_ot:
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, p_gauss, opt)
                        # q type: torch.Tensor, size: ([num_frames, num_actions])

                    else:
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, None, opt)
                    
                    
                    if (i + (epoch * len(train_loader))) % 500 == 0:
                        img_finalq= plot_confusion_matrix(q.clone().detach().cpu().numpy())
                        img_protos = plot_confusion_matrix(proto_scores.clone().detach().cpu().numpy())
                        prototypes = model.get_prototypes()
                        dists = compute_euclidean_dist(prototypes, prototypes)
                        img_dists = plot_confusion_matrix(dists.detach().cpu().numpy())

                        writer.add_image("Q Matrix", img_finalq, i + (epoch * len(train_loader)))
                        writer.add_image("Dot-Product Matrix", img_protos,  i + (epoch * len(train_loader)))
                        writer.add_image("Prototype Dists", img_dists, i + (epoch * len(train_loader)))
                        #np.save(f, q.clone().detach().cpu().numpy())
            
                if opt.use_transformer:
                    proto_probs = F.softmax(proto_scores / opt.temperature, dim=1)
                
                else:
                    proto_probs = F.softmax(proto_scores/opt.temperature) 
        
                 
                if i + (epoch * len(train_loader)) % 500 == 0:
                    with torch.no_grad():
                        img = plot_confusion_matrix(proto_probs.clone().detach().cpu().numpy())
                        writer.add_image("P Matrix", img, i + (epoch * len(train_loader)))

            
                proto_probs = torch.clamp(proto_probs, min= 1e-30, max=1)

                ################ ADDED #################
                if opt.use_transformer:
                    proto_loss = torch.mean(torch.sum(q * torch.log(proto_probs + 1e-8), dim=1))

                else:
                ##########################################
                    proto_loss = torch.mean(torch.sum(q * torch.log(proto_probs), dim = 1))
            
            loss_tcn =  tcn_loss(embs)
            loss_values = 0 

            if opt.tcn_loss:
                loss_values += loss_tcn 
            
            if opt.time_loss:
                loss_values = loss(output.squeeze(), labels.squeeze())  #+ loss_tcn(embeddings) #loss_tcn(embeddings) 
            
            if learn_prototype and (i + (epoch * len(train_loader))) >= opt.freeze_proto_loss:
               loss_values -= proto_loss
               
            c_losses.update(proto_loss.item(), 1)
            tcn_losses.update(loss_tcn.item(), 1)

            losses.update(loss_values.item(), features.size(0))
            

            optimizer.zero_grad()

            if not opt.use_transformer:
                loss_values.backward() # Optimize with respect to the loss_values
            
            else:
                if torch.isfinite(loss_values).all():
                    loss_values.backward() 
            
                else:
                    print("Loss is NaN or Inf")
                    optimizer.zero_grad() 
                    continue  # Skip this update to avoid corrupting model weights


            if i + (epoch * len(train_loader)) < opt.freeze_iters:

                for name, p in model.named_parameters():
                   
                    if "prototype" in name:
                        p.grad = None 
            
            ############# ADDED  ##############
            if opt.use_transformer:
                #  clip gradients to prevent exploding gradients which can cause NaN values:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    print("NaN or Inf found in gradients, skipping optimizer step")
                    optimizer.zero_grad()
                    continue  # Skip this update
                else:
                    optimizer.step()
                    total_loss += loss_values.item() # For early stopping
            ###################################################

            else:
                optimizer.step()
                total_loss += loss_values.item() # For early stopping

            batch_time.update(time.time() - end)
            end = time.time()

            writer.add_scalar("Loss/task_loss", loss_values, i + (epoch * len(train_loader)))
            if learn_prototype:
                writer.add_scalar("Loss/cluster_loss", -proto_loss, i + (epoch * len(train_loader)))
                writer.add_scalar("Loss/tcn_loss", loss_tcn, i + (epoch * len(train_loader)))

            if i % 20 == 0 and i:
                
                
                if not learn_prototype:
                    #print("HERE")
                
                    logger.debug('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))
                else:
                    print("HEREEEEE")
                    logger.debug('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Cluster Loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t'
                                'TCN Loss {tcn_loss.val: .4f} ({tcn_loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, c_loss = c_losses, tcn_loss = tcn_losses))

            #################### ADDED #################################
            if learn_prototype and opt.apply_permutation_aware_prior:
                # Prepare and save estimated transcripts for the current batch
                num_frames = features.shape[0]
                num_actions = proto_scores.shape[1]

                # Estimate transcripts from the frame-level pseudo-label codes Qf
                if opt.estimate_transcript_method == "basic":
                    estimated_transcripts = estimate_transcripts_basic(q, num_actions)
                
                elif opt.estimate_transcript_method == "improved":
                    peaks_distance = num_frames*0.6
                    estimated_transcripts = estimate_transcripts(q, num_actions, peaks_distance, 0.5)
                
                #print(estimated_transcripts)

                estimated_transcripts_all = prepare_transcripts(opt, estimated_transcripts, estimated_transcripts_all, video_ids)

            ###############################################################

        # Print weights after the first epoch
        if epoch == 0:
            print("Model weights after the first epoch:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.data}")
        
        average_loss = total_loss / len(train_loader)

        ############## ADDED ######################
        # Early stopping check (only after some epochs)
        if opt.early_stopping and epoch >= opt.early_stop_min_epochs:
            if early_stopper(average_loss):
                print("Early stopping triggered")
                early_stopped = True
                break
        ##############################################

        ################# ADDED #############
        # After each epoch append the epoch number and the average losses (average loss across batches in the current epoch)
        with open(final_losses_path, 'a') as file:
            file.write(f'Epoch {epoch + 1}: {losses.avg:.4f}\n') # Task losses

        with open(cluster_losses_path, 'a') as file:
            file.write(f'Epoch {epoch + 1}: {c_losses.avg:.4f}\n')

        with open(TCN_losses_path, 'a') as file:
            file.write(f'Epoch {epoch + 1}: {tcn_losses.avg:.4f}\n')
        ############################

        logger.debug('loss: %f' % losses.avg)
     
        losses.reset()

    
    f.close()

    ####### ADDED #################
    # Save the estimated transcripts after training or early stopping
    if learn_prototype and opt.apply_permutation_aware_prior:
        save_transcripts(opt, estimated_transcripts_all, corpus)

    # Save the last Q computed
    """q_values = q.clone().detach().cpu().numpy()
    if q_values is not None:
        filename = 'q_values_last_video_last_epoch_' + opt.description + '.npy'
        np.save(join("additional_outputs", filename), q_values)
        print("q values from the last epoch saved.")"""
    
    # Save last fixed-order prior distribution
    tensorboard_dir = '/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs'
    filename = 'p_gauss_Ma.npy'
    path = os.path.join(tensorboard_dir, opt.description, filename)
    np.save(path, p_gauss)
    
    #filename = 'p_gauss_Ma_' + opt.description + '.npy'
    #np.save(join("additional_outputs", filename), p_gauss)
    #################################

    opt.resume_str = join(opt.tensorboard_dir, 'models',
                          '%s.pth.tar' % opt.log_str)
    if save:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        if opt.global_pipe:
            dir_check(join(opt.dataset_root, 'models', 'global'))
            opt.resume_str = join( opt.tensorboard_dir, 'models', 'global',
                                  '%s.pth.tar' % opt.log_str)
        else:
            dir_check(join(opt.tensorboard_dir, 'models'))
            save_params(opt, join(opt.tensorboard_dir))
        torch.save(save_dict, opt.resume_str)

    tracker.stop() 
    return model

def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

        curr_sum = torch.sum(Q, dim=1)
        # dist.all_reduce(curr_sum)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
            # dist.all_reduce(curr_sum)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def load_model(opt):
    if opt.loaded_model_name:
        if opt.global_pipe:
            resume_str = opt.loaded_model_name
        else:
            resume_str = opt.loaded_model_name #% opt.subaction
        # resume_str = opt.resume_str
    else:
        resume_str = opt.log_str + '.pth.tar'
    # opt.loaded_model_name = resume_str
    if opt.device == 'cpu':
        checkpoint = torch.load(join(opt.dataset_root, 'models',
                                     '%s' % resume_str),
                                map_location='cpu')
    else:
        checkpoint = torch.load(join(opt.tensorboard_dir, 'models',
                                 '%s' % resume_str))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s' % resume_str)
    return checkpoint


def get_cost_matrix(batch_size, num_videos, num_videos_dataset, num_clusters, sigma):

    cost_matrix = generate_matrix(int(batch_size/num_videos_dataset), num_clusters)
    cost_matrix = np.vstack([cost_matrix] * num_videos)
    p_gauss = gaussian(cost_matrix, sigma = sigma)

    return p_gauss


def cost(i, j, n, m):

  return ((i - (j/m) *n)/n)**2


def cost_paper(i, j, n, m):
    return ((abs(i/n - j/m))/(np.sqrt((1/n**2) + (1/m**2))))**2


def gaussian(cost, sigma):
    #print("Value of sigma: {}".format(opt.sigma))
    return (1/(sigma * 2*3.142))*(np.exp(-cost/(2*(sigma**2))))

def generate_matrix(num_elements, num_clusters):

    cost_matrix = np.zeros((num_elements, num_clusters))

    for i in range(num_elements):
        for j in range(num_clusters):

            cost_matrix[i][j] = cost_paper(i, j, num_elements, num_clusters)

    return cost_matrix

def plot_confusion_matrix(q):

    fig, ax = plt.subplots(nrows=1)
    sns.heatmap(q, ax = ax)
    image = plot_to_image(fig)

    return image

def compute_euclidean_dist(embeddings, prototypes):

    with torch.no_grad():

        dists = torch.sum(embeddings**2, dim = 1).view(-1, 1) + torch.sum(prototypes**2, dim = 1).view(1, -1) -2 * torch.matmul(embeddings, torch.transpose(prototypes, 0, 1)) 
        return dists

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = imread(buf)

  # Add the batch dimension
  #image = tf.expand_dims(image, 0)
  return image.transpose(2, 0, 1)


def generate_optimal_transport_labels(proto_scores, epsilon, p_gauss, opt):
    if opt.use_transformer:
        max_proto_scores = torch.max(proto_scores, dim=1, keepdim=True)[0]
        q = (proto_scores - max_proto_scores) / epsilon

    else:
        q = proto_scores/epsilon
    
    q =  torch.exp(q)
    if p_gauss is not None:
        q = q * torch.from_numpy(p_gauss).cuda()
    q = q.t()
    q =  distributed_sinkhorn(q, 3)

    return q


########### ADDED ###########################################

def estimate_transcripts_basic(q, num_actions):
    """ Takes the TENSOR q and returns a sorted list of actions by frame index """
    # For each j-th action, find the i-th frame where Q_{f}^{ij} has the maximum assignment probability 
    # (along the j-th column), which results in an action-frame pair (j,i).

    # Find the frame index with the highest probability for each action
    frame_indices = torch.argmax(q, dim=0)  # Finding max along rows, results in [num_actions]
    
    # Create action-frame pairs
    action_frame_pairs = [(i + 1, int(frame_indices[i])) for i in range(num_actions)]
    
    # Sort all action-frame pairs by their frame indexes
    action_frame_pairs.sort(key=lambda x: x[1])  # Sort by frame index

    # The resulting temporally sorted list of actions is our estimates transcript T
    return action_frame_pairs


def estimate_transcripts(q, num_actions, min_separation=50, max_relative_difference=0.7):
    """
    Estimate transcripts allowing actions to appear multiple times based on significant probability peaks.

    Args:
    q: Tensor of shape (num_frames, num_actions), representing the probability that a frame is assigned to a certain action
    num_actions: Number of actions.
    min_separation: Minimum separation between peaks for the same action.
    max_relative_difference: Maximum relative difference for considering secondary peaks compared to the primary peak.

    Returns:
    Transcript: list of (action, frame) pairs representing the estimated transcript.
    """
    q = q.cpu()

    action_frame_pairs = []

    for action_index in range(num_actions):
        probabilities = q[:, action_index].numpy()

        # Find primary peak (highest probability frame)
        primary_peak_index = np.argmax(probabilities)
        primary_peak_value = probabilities[primary_peak_index]

        action_frame_pairs.append((int(action_index + 1), int(primary_peak_index), primary_peak_value))
        #print(f"Action {action_index + 1}: Primary peak at frame {primary_peak_index} with value {primary_peak_value}")

        # Find the second highest peak
        secondary_peak_index = -1
        secondary_peak_value = -1
        for i, value in enumerate(probabilities):
            if i != primary_peak_index and value > secondary_peak_value:
                secondary_peak_index = i
                secondary_peak_value = value

        # Check the conditions for considering the secondary peak
        if secondary_peak_index != -1:
            distance = abs(primary_peak_index - secondary_peak_index)
            relative_difference = secondary_peak_value / primary_peak_value
            #print(f"Action {action_index + 1}: Secondary peak at frame {secondary_peak_index} with value {secondary_peak_value}, distance {distance}, relative difference {relative_difference}")

            if distance >= min_separation and relative_difference >= max_relative_difference:
                action_frame_pairs.append((int(action_index + 1), int(secondary_peak_index), secondary_peak_value))
                #print(f"Action {action_index + 1}: Secondary peak added at frame {secondary_peak_index}")


    # Sort by probability value in descending order (to keep the top ones)
    action_frame_pairs.sort(key=lambda x: x[2], reverse=True)

    print("action_frame inicial: ", action_frame_pairs)

    # Keep only the top num_actions pairs
    action_frame_pairs = action_frame_pairs[:num_actions]

    # Sort by frame index to maintain temporal order
    action_frame_pairs.sort(key=lambda x: x[1])

    # Return only the action and frame pairs
    return [(action, frame) for action, frame, _ in action_frame_pairs]

