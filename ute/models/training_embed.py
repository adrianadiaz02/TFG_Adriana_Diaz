#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

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
#############################################################33


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
        early_stopper = EarlyStopping(patience=15, verbose=True, delta=0.05)
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

    #################### AÑADIDO ######################
    # Files for the 3 losses (cluster, tcn, and final losses)
    TCN_losses_path = join(opt.tensorboard_dir, 'TCN_Losses.txt')

    cluster_losses_path = join(opt.tensorboard_dir, 'Cluster_Losses.txt')
    final_losses_path = join(opt.tensorboard_dir, 'Final_Losses.txt')
    ###################################################


    logger.debug('epochs: %s', epochs)
    f = open("test_q_distribution.npy", "wb")

    #############
    torch.autograd.set_detect_anomaly(True)
    #############

    ############# AÑADIDO ###################
    if opt.apply_permutation_aware_prior:
        estimated_transcripts_all = {}
    #########################################

    for epoch in range(epochs):
        # model.cuda()
        model.to(opt.device)
        model.train()

        total_loss = 0 # AÑADIDO

        logger.debug('Epoch # %d' % epoch)
      
        end = time.time()
      
        for i, (features, labels, video_ids, video_names) in enumerate(train_loader): # videos_ids AÑADIDO

            #print(f"Processing {video_ids}")

            num_videos = features.shape[0] 
            #print("Number of videos:", num_videos) # >> 2 o 1(Number of videos in a batch)

            if not opt.use_transformer:
                #print("Features shape: ", features.shape) # >> [2, 128, 64] (SIEMPRE), 2 videos, batch size 256
                features = torch.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
                #print("Features shape afer reshape: ", features.shape) # >> [256, 64] 
                            
            else:
                features = features.transpose(0, 1)  # Transpose to [sequence_length, batch_size, feature_dim]
                print("shape ", i, video_ids, features.shape)
                print(f"{video_ids} - max: {features.max().item()}, min: {features.min().item()}, mean: {features.mean().item()}")

                if not torch.isfinite(features).all():
                    print("Features contain NaN or Inf", i, video_ids)

            # obs: Features does NOT contain NaN or Inf


            labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1], labels.shape[2]))

            if opt.use_transformer:
                if not torch.isfinite(labels).all():
                    print("Labels contain NaN or Inf", i, video_ids)
            
            # obs: Labels does NOT contain NaN or Inf

            data_time.update(time.time() - end)
            features = features.float()
            labels = labels.float().to(opt.device)
            
            if opt.device == 'cuda':
                features = features.cuda(non_blocking=True)

            output, proto_scores, embs = model(features)
            
            if opt.use_transformer:
                print("output:")
                print(f"{video_ids} - max: {output.max().item()}, min: {output.min().item()}, mean: {output.mean().item()}")

                print("proto_scores:")
                print(f"{video_ids} - max: {proto_scores.max().item()}, min: {proto_scores.min().item()}, mean: {proto_scores.mean().item()}")

                print("embs:")
                print(f"{video_ids} - max: {embs.max().item()}, min: {embs.min().item()}, mean: {embs.mean().item()}")

            #print("output: ", output.shape) # >> [512, 1]
            #print("proto scores: ", proto_scores.shape) # >> [512, 7]
            #print("embs: ", embs.shape) # >> [512, 40]


            if learn_prototype:
                with torch.no_grad():

                    #compute q
                    p_gauss = get_cost_matrix(batch_size = opt.batch_size, num_videos = num_videos, num_videos_dataset = opt.num_videos \
                    ,sigma = opt.sigma, num_clusters = proto_scores.shape[1]) # Prior distribution, mantiene fixed order the los clusters
                    #print("p_gauss: ", p_gauss.shape) # -> p_gauss:  (512, 7)  

                    if opt.apply_temporal_ot:
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, p_gauss)
                        # q type: torch.Tensor, size: ([num_frames, num_actions])

                    else:
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, None)
                    
                    
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
                    #max_scores = torch.max(proto_scores, dim=1, keepdim=True)[0]
                    proto_probs = F.softmax((proto_scores) / opt.temperature, dim=1)
                    proto_probs = torch.clamp(proto_probs, min=1e-8, max=1-1e-8) # SACAR????

                
                else:
                    proto_probs = F.softmax(proto_scores/opt.temperature) 
                
                """if torch.isfinite(proto_scores).all():
                    print("proto_scores contains NaN or Inf")
                if torch.isfinite(proto_probs).all():
                    print("proto_probs contains NaN or Inf")"""
        
                 
                if i + (epoch * len(train_loader)) % 500 == 0:
                    with torch.no_grad():
                        img = plot_confusion_matrix(proto_probs.clone().detach().cpu().numpy())
                        writer.add_image("P Matrix", img, i + (epoch * len(train_loader)))

                
                ################ AÑADIDO #################
                if opt.use_transformer:
                    print("q", q)
                    print("proto_probs: ")
                    print(f"{video_ids} - max: {proto_probs.max().item()}, min: {proto_probs.min().item()}, mean: {proto_probs.mean().item()}")

                    print("torch.log(proto_probs): ")
                    print(f"{video_ids} - max: {(torch.log(proto_probs + 1e-8)).max().item()}, min: {(torch.log(proto_probs + 1e-8)).min().item()}, mean: {(torch.log(proto_probs + 1e-8)).mean().item()}")

                    print("q * torch.log(proto_probs + 1e-8):")
                    print(f"{video_ids} - max: {(q * torch.log(proto_probs + 1e-8)).max().item()}, min: {(q * torch.log(proto_probs + 1e-8)).min().item()}, mean: {(q * torch.log(proto_probs + 1e-8)).mean().item()}")

                    proto_loss = torch.mean(torch.sum(q * torch.log(proto_probs + 1e-8), dim=1))
                    #proto_loss = torch.mean(torch.sum(q * torch.nn.functional.log_softmax(proto_probs, dim=1), dim=1))
                    #proto_loss = torch.mean(torch.sum(q * torch.nn.functional.log_softmax(proto_scores+ 1e-8, dim=1), dim=1))
                    print("proto loss: ", proto_loss)
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
                loss_values.backward() # OPTIMIZA A PARTIR DE LOSS_VALUES
            
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
            
            ############# AÑADIDO p/ transformer ##############
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

            #################### AÑADIDO #################################
            if learn_prototype and opt.apply_permutation_aware_prior:
                # Additional condition to only calculate during the last epoch
                if epoch == epochs - 1:
                    num_frames = features.shape[0]
                    num_actions = proto_scores.shape[1]

                    # Estimate transcripts from the frame-level pseudo-label codes Qf
                    estimated_transcripts = estimate_transcripts_basic(q, num_actions)
                    #estimated_transcripts = estimate_transcripts(q, num_actions, 0.7)
                    #print(estimated_transcripts)

                    """# Compute Qs from T
                    #qs = compute_qs_from_transcripts(estimated_transcripts, num_actions, num_frames)
                    
                    # Compute prior distribution Mt that imposes the permutation-aware transcript 
                    cost_matrix = create_cost_matrix(num_frames, num_actions, estimated_transcripts)
                    Mt = gaussian_transform(cost_matrix, 6) # Mt type: numpy.ndarray, size: (128, 7)...
                    
                    # Compute Qa mith Mt. And use Qa for prototype loss calculation
                    q = generate_optimal_transport_labels(proto_scores, opt.epsilon, Mt)
                    # q type: torch.Tensor, size: ([128, 7]"""

                    # Store the transcripts using the video name corresponding to the video_id as key
                    mapping_path =  f'video_id_mapping_{opt.subaction}.json'
                    with open(mapping_path, 'r') as file:
                        video_name_mapping = json.load(file)

                    for video_id in video_ids:
                        if video_id in video_name_mapping:
                            # instead of saving with the video_id, save it with its corresponding name
                            descriptive_name = video_name_mapping[video_id]  # Get the corresponding descriptive name
                            estimated_transcripts_all[descriptive_name] = estimated_transcripts
                        else:
                            logger.error(f"Video ID {video_id} not found in video_name_mapping")



            ###############################################################

        # Print weights after the first epoch
        if epoch == 0:
            print("Model weights after the first epoch:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.data}")
        
        average_loss = total_loss / len(train_loader)

        ############## AÑADIDO ######################
        # Early stopping check
        if opt.early_stopping:
            if early_stopper(average_loss):
                print("Early stopping triggered")
                early_stopped = True
                break
        ##############################################

        ################# AÑADIDO #############
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

    ###### AÑADIDO - PERMUTATION AWARE ##############
    # After the training loop ends, save the transcripts
    if opt.apply_permutation_aware_prior:
        # Define the filename incorporating subaction name
        filename = f'estimated_transcripts_{opt.subaction}.json'
        try:
            # Save the estimated transcripts for each video
            print("GUARDANDO TRANSCRIPT")
            with open(filename, 'w') as file:
                json.dump(estimated_transcripts_all, file)
            logger.info(f"Successfully saved estimated transcripts to {filename}.")
        
        except IOError as e:
            logger.error(f"Failed to save estimated transcripts: {str(e)}")
        
        corpus.update_transcripts(estimated_transcripts_all)

    
    f.close()

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

def load_model():
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


def generate_optimal_transport_labels(proto_scores, epsilon, p_gauss):
    
    q = proto_scores/epsilon
    q =  torch.exp(q)
    if p_gauss is not None:
        q = q * torch.from_numpy(p_gauss).cuda()
    q = q.t()
    q =  distributed_sinkhorn(q, 3)

    return q


########### AÑADIDO ###########################################

def estimate_transcripts_basic(q, num_actions):
    """ Takes the TENSOR q and returns a sorted list of actions by frame index """
    # For each j-th action, find the i-th frame where Q_{f}^{ij} has the maximum assignment probability (along the j-th column) 
    #   --> action-frame pair (j,i)  

    # Find the frame index with the highest probability for each action
    frame_indices = torch.argmax(q, dim=0)  # Finding max along rows, results in [num_actions]
    
    # Create action-frame pairs
    action_frame_pairs = [(i + 1, int(frame_indices[i])) for i in range(num_actions)]
    
    # Sort all action-frame pairs by their frame indexes
    action_frame_pairs.sort(key=lambda x: x[1])  # Sort by frame index

    # The resulting temporally sorted list of actions is our estimates transcript T
    return action_frame_pairs


def estimate_transcripts(q, num_actions, threshold=0.8, separation=50, relative_height=0.7):
    """
    Estimate transcripts allowing actions to appear multiple times based on significant probability peaks.

    Args:
    q: Tensor of shape (num_frames, num_actions), representing the probability that a frame is assigned to a certain action
    num_actions: Number of actions.
    threshold: Minimum height (probability) for peaks.
    separation: Minimum separation between peaks for the same action.
    relative_height: Relative height for considering secondary peaks compared to the primary peak.

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
        
        if primary_peak_value >= threshold: # permitimos que hayan acciones no asignadas?
            action_frame_pairs.append((int(action_index + 1), int(primary_peak_index)))

            # Find additional peaks
            peaks, properties = find_peaks(probabilities, height=primary_peak_value * relative_height, distance=separation)

            # Select only the most significant secondary peak
            if len(peaks) > 0:
                secondary_peak_index = peaks[np.argmax(properties['peak_heights'])]
                if secondary_peak_index != primary_peak_index:  # Avoid duplicate peaks
                    action_frame_pairs.append((int(action_index + 1), int(secondary_peak_index)))

    # Sort by frame index to maintain temporal order
    action_frame_pairs.sort(key=lambda x: x[1])

    return action_frame_pairs


""" Computes Qs based on the estimated transcripts """
def compute_qs_from_transcripts(transcript, num_actions, num_frames):
    # set Q_{s}^{ij} = 1 if the i-th position in T contains the j-th action
    # set Q_{s}^{ij} = 0 otherwise 

    qs = torch.zeros(num_frames, num_actions)
    for idx, (frame, action) in enumerate(transcript):
        qs[frame, action] = 1
    return qs

""" Generate a vertical Gaussian distribution for a specific frame. """
def vertical_gaussian(num_frames, center_frame, sigma):
    frame_range = np.arange(num_frames)
    gaussian_distribution = np.exp(-0.5 * ((frame_range - center_frame) ** 2) / sigma**2)
    gaussian_distribution /= np.sum(gaussian_distribution)  # Normalize the distribution
    return gaussian_distribution

""" Generate a matrix with vertical Gaussian distributions based on the chronological order of actions. """
def generate_matrix2(num_frames, transcript, sigma):
    # ! Transcript should be sorted by frame number

    Mt = np.zeros((num_frames, len(transcript)))
    
    for idx, (action, frame) in enumerate(transcript):
        gaussian_distribution = vertical_gaussian(num_frames, frame, sigma)
        Mt[:, action-1] = gaussian_distribution
    
    return Mt

def plot_prior(Mt, tensorboard_dir, description,):
    # Create the full path for the output directory if it doesn't exist
    output_dir = os.path.join(tensorboard_dir, description)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "heatmap_Mt.png")

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(Mt, annot=False, cmap="viridis")
    plt.title("Last Mt Distribution of epoch 3999")
    plt.xlabel("Action Index")
    plt.ylabel("Frame Number")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y-axis labels to 0 degrees

    plt.savefig(output_file, format='png', dpi=300)
    plt.close()
    logger.debug(f'Saved heatmap to {output_file}')

def create_cost_matrix(num_frames, num_clusters, transcript):
    # Initialize the cost matrix with high values
    cost_matrix = np.full((num_frames, num_clusters), np.inf)

    # Create a dictionary from the transcript for quick access
    action_to_frame = {action: frame for action, frame in transcript}

    # Iterate through each frame
    for i in range(num_frames):
        # Iterate through each cluster (action)
        for j in range(1, num_clusters + 1):
            # If the action is in the transcript, compute the cost based on the distance
            if j in action_to_frame:
                assigned_frame = action_to_frame[j]
                # The cost is the absolute distance between the current frame and the assigned frame
                cost_matrix[i, j - 1] = abs(i - assigned_frame)

    return cost_matrix

def gaussian_transform(cost_matrix, sigma):
    # Apply the Gaussian function to the cost matrix to create the probability matrix
    return (1/(sigma * 2*3.142))*(np.exp(-cost_matrix/(2*(sigma**2))))

# Just for testing the computation of Mt
def test1():
    transcript = [
        (1, 1),  # Action 1 at Frame 1
        (2, 56),  # Action 2 at Frame 56
        (3, 80),   # Action 3 at Frame 80
        (5, 96),
        (4, 118),
        (9, 133),
        (6, 141),
        (7, 220),
        (8, 241),
        (10, 292),
        (11, 375)
    ]

    num_frames = 400
    num_actions = 11
    sigma = 6

    #Mt = generate_matrix2(num_frames, transcript, sigma)

    cost_matrix = create_cost_matrix(num_frames, num_actions, transcript)
    cost_matrix = gaussian_transform(cost_matrix, sigma)

    Mt = cost_matrix

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(Mt, annot=False, cmap="viridis")
    plt.title("Permutation Aware Prior Distribution ")
    plt.xlabel("Action Index")
    plt.ylabel("Frame Number")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y-axis labels to 0 degrees

    plt.show()
    print("aqui")

    plt.savefig("heatmapMt_sigma6.png", format='png', dpi=300) 
    plt.close()

def test2():
    # Simulate a probability matrix q
    num_frames = 10
    num_actions = 5
    q = [
        [0.14964014, 0.8491676, 0.13654268, 0.92348254, 0.15755248],
        [0.9082902, 0.97152746, 0.55360454, 0.45107824, 0.42410356],
        [0.3619051, 0.8833675, 0.62355417, 0.14134067, 0.11591011],
        [0.962852, 0.85274047, 0.14426965, 0.24881577, 0.6058707],
        [0.042896986, 0.8406826, 0.896343, 0.34544802, 0.63865143],
        [0.44632727, 0.032742202, 0.082745135, 0.93800163, 0.20366102],
        [0.69847405, 0.5515126, 0.29898065, 0.7774229, 0.366472],
        [0.11915469, 0.8740935, 0.8785842, 0.34026724, 0.3189271],
        [0.37987816, 0.04573065, 0.669469, 0.7632588, 0.6572629],
        [0.07546544, 0.22469765, 0.90228134, 0.7610668, 0.31091696]
    ]
    # Convert q to a NumPy array and then tensor
    q_array = np.array(q)
    q_tensor = torch.tensor(q_array, dtype=torch.float32)

    # Estimate transcripts
    transcripts = estimate_transcripts(q_tensor, num_actions)
    
    print("Sorted Action-Frame Pairs:")
    for action, frame in transcripts:
        print(f"Action {action} is most likely at frame {frame}")

def test3():
    num_frames = 400
    num_actions = 11
    sigma = 1
    p_gauss = get_cost_matrix(batch_size = 1, num_videos = 1, num_videos_dataset = 1 \
                        ,sigma = sigma, num_clusters = num_actions)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(p_gauss, annot=False, cmap="viridis")
    plt.title("Permutation Aware Prior Distribution ")
    plt.xlabel("Action Index")
    plt.ylabel("Frame Number")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y-axis labels to 0 degrees

    plt.show()

    plt.savefig("heatmapMa.png", format='png', dpi=300) 
    plt.close()


#################################################################