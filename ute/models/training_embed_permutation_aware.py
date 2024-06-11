#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training']
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


############# AÑADIDO ######################
class EarlyStopping:
    """Early stops the training if the loss doesn't improve or changes minimally after a given patience."""
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

# Load video ID to descriptive name mapping
def load_video_name_mapping(opt):
    mapping_path = os.path.join("video_id_mappings", f'video_id_mapping_{opt.subaction}.json')
    with open(mapping_path, 'r') as file:
        video_name_mapping = json.load(file)
    return video_name_mapping
   
#####################################

def training_permutation_aware(train_loader, epochs, save, corpus, transcripts, **kwargs):
    """Training pipeline for embedding.
    Different from the other embedding training, this uses the transcript estimated from the previous training
    to improve the embeddings.

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

    #################### ADDED ######################
    # Load video ID to descriptive name mapping
    video_name_mapping = load_video_name_mapping(opt)

    # File for the losse
    final_losses_path = join(opt.tensorboard_dir, 'Final_Losses_Second_Train.txt')
    ###################################################

    logger.debug('epochs: %s', epochs)
    f = open("test_q_distribution.npy", "wb")

    #############
    torch.autograd.set_detect_anomaly(True)

    # Start emissions tracking
    tensorboard_dir = '/home/usuaris/imatge/adriana.diaz/TOT-CVPR22-main/runs'
    path = os.path.join(tensorboard_dir, opt.description)

    tracker = EmissionsTracker(output_dir=path, project_name="second_regression_training")
    tracker.start()

    #########################################

    for epoch in range(epochs):
        # model.cuda()
        model.to(opt.device)
        model.train()

        logger.debug('Epoch # %d' % epoch)
      
        end = time.time()

        total_loss = 0 # For early stopping
      
        for i, (features, labels, video_ids, video_names) in enumerate(train_loader): # videos_ids y video_names AÑADIDO
            
            num_videos = features.shape[0] 
            num_frames = features.shape[1]

            # Get the estimated transcript corresponding to the video
            for video_name in video_names:
                if video_name in transcripts:
                    transcript = transcripts[video_name]
                    
                    # Normalize transcript labels to start from 0
                    transcript = [(label - 1, start) for label, start in transcript]
                    #print(f"Transcript for {video_name}: {transcript}")
                else:
                    print(f"No transcript found for {video_name}")

            
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
                features = torch.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))                            
                labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1], labels.shape[2]))
            

            data_time.update(time.time() - end)
            features = features.float()
            labels = labels.float().to(opt.device)


            if opt.device == 'cuda':
                features = features.cuda(non_blocking=True)

            output, proto_scores, embs = model(features)

            if learn_prototype:
                with torch.no_grad():

                    # Compute the corresponding Permutation-aware prior distribution (Mt)
                    num_actions = len(transcript)

                    # Get the reference fixed-order prior
                    p_gauss_path = join(tensorboard_dir, opt.description, 'p_gauss_Ma.npy')
                    p_gauss = np.load(p_gauss_path)
        
                    # Compute the permutation-aware prior M_t
                    num_clusters = proto_scores.shape[1]
                    M_t = compute_permutation_aware_prior(transcript, num_clusters, num_frames, opt.sigma)

                    # Normalize the permutation-aware prior to match the reference prior
                    M_t = normalize_prior(M_t, p_gauss)

                    if opt.apply_temporal_ot:
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, M_t, opt)
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
            
                
                proto_probs = F.softmax(proto_scores/opt.temperature, dim=1) 
        
                 
                if i + (epoch * len(train_loader)) % 500 == 0:
                    with torch.no_grad():
                        img = plot_confusion_matrix(proto_probs.clone().detach().cpu().numpy())
                        writer.add_image("P Matrix", img, i + (epoch * len(train_loader)))

            
                proto_probs = torch.clamp(proto_probs, min= 1e-30, max=1)

                ################ AÑADIDO #################
                if torch.isfinite(proto_probs).all() and torch.isfinite(q).all():
                    proto_loss = torch.mean(torch.sum(q * torch.log(proto_probs + 1e-8), dim=1))
                else:
                    if not torch.isfinite(proto_probs).all():
                        raise ValueError("NaN detected in proto_probs")
                    if not torch.isfinite(q).all():
                        raise ValueError("NaN detected in q")

            ############### ADDED #########################
            # save the last Qs computed
            """if epoch == epochs - 1:
                # Get the video name for this batch
                batch_video_name = video_names[0]  # Assuming all videos in the batch have the same name

                # Save Q matrix for this video
                q_values = q.clone().detach().cpu().numpy()
                file_name = f'Q_{batch_video_name}.npy'
                np.save(join("additional_outputs", 'Q_values2', file_name), q_values)
                print("Q values from the last epoch saved.")

                # Save the prior distribution matrix for this video
                file_name = f'Mt_{batch_video_name}.npy'
                np.save(join("additional_outputs", 'Mt_values2', file_name), M_t)
                print("Mt values from the last epoch saved.")"""

            ################################################
             
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

            if torch.isfinite(loss_values).all():
                loss_values.backward() # OPTIMIZA A PARTIR DE LOSS_VALUES
        
            else:
                print("Loss is NaN or Inf")
                optimizer.zero_grad() 
                continue  # Skip this update to avoid corrupting model weights


            if i + (epoch * len(train_loader)) < opt.freeze_iters:

                for name, p in model.named_parameters():
                   
                    if "prototype" in name:
                        p.grad = None 
            
            ############# AÑADIDO  ##############
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


        # Print weights after the first epoch
        if epoch == 0:
            print("Model weights after the first epoch:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.data}")
        
        average_loss = total_loss / len(train_loader)
        if average_loss != losses.avg:
            print("average_loss: ", average_loss)
            print("losses.avg: ", losses.avg)
        #average_loss = losses.avg

        ############## ADDED ######################
        # Early stopping check (only after some epochs)
        if opt.early_stopping and epoch >= opt.early_stop_min_epochs:
            if early_stopper(average_loss):
                print("Early stopping triggered")
                early_stopped = True
                break
        ##############################################

        ################# AÑADIDO #############
        # After each epoch append the epoch number and the average losses (average loss across batches in the current epoch)
        with open(final_losses_path, 'a') as file:
            file.write(f'Epoch {epoch + 1}: {losses.avg:.4f}\n') 

        ############################

        logger.debug('loss: %f' % losses.avg)
     
        losses.reset()

    
    f.close()


    # Save last prior distribution
    #filename = 'Mt2_' + opt.description + '.npy'
    #np.save(join("additional_outputs", filename), M_t)


    """opt.resume_str = join(opt.tensorboard_dir, 'models',
                          '%s.pth.tar' % opt.log_str)"""
    
    opt.resume_str = join(opt.tensorboard_dir, 'models', 'model_second_phase.pth.tar')

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


def generate_optimal_transport_labels(proto_scores, epsilon, prior, opt):
    max_proto_scores = torch.max(proto_scores, dim=1, keepdim=True)[0]
    q = (proto_scores - max_proto_scores) / epsilon
    
    q =  torch.exp(q)
    if prior is not None:
        q = q * torch.from_numpy(prior).cuda()
    q = q.t()
    q =  distributed_sinkhorn(q, 3)

    return q


########### ADDED ###########################################
def compute_permutation_aware_prior(transcript, num_clusters, num_frames, sigma):
    """Compute the permutation-aware prior for the current batch using the estimated transcripts."""
    cost_matrix = np.zeros((num_frames, num_clusters))
    
    # Calculate midpoints of the action segments
    action_positions = {}
    for idx, (action, start_frame) in enumerate(transcript):
        if idx < len(transcript) - 1:
            next_start_frame = transcript[idx + 1][1]
        else:
            next_start_frame = num_frames  # Assume the last action goes until the end of the frames
        
        mid_point = (start_frame + next_start_frame) // 2
        action_positions[action] = mid_point

    for j in range(num_clusters):
        if j + 1 in action_positions:
            frame_position = action_positions[j + 1]
            for i in range(num_frames):
                cost_matrix[i, j] = cost_paper(i, frame_position, num_frames, num_clusters)

    p_permutation_aware = gaussian(cost_matrix, sigma)
    return p_permutation_aware


def normalize_prior(prior, reference_prior):
    M_t = prior * (reference_prior.sum() / prior.sum())
    M_t += 0.0001 # Add a small constant value
    return M_t 