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


def training(train_loader, epochs, save, **kwargs):
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

    ######################## AÑADIDO ##################
    """
    # Initialize variables to track the best model
    lowest_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    from ute.corpus import Corpus
    corpus = Corpus(subaction=opt.subaction, opt = opt) # loads all videos, features, and gt
    """

    # Files for the 4 losses (task, cluster, tcn, and final losses)
    task_losses_path = join(opt.tensorboard_dir, 'Task_Losses.txt')
    TCN_losses_path = join(opt.tensorboard_dir, 'TCN_Losses.txt')

    cluster_losses_path = join(opt.tensorboard_dir, 'Cluster_Losses.txt')
    final_losses_path = join(opt.tensorboard_dir, 'Final_Losses.txt')
    ###################################################

    logger.debug('epochs: %s', epochs)
    f = open("test_q_distribution.npy", "wb")
    for epoch in range(epochs):
        # model.cuda()
        model.to(opt.device)
        model.train()

        logger.debug('Epoch # %d' % epoch)
      
        end = time.time()
      
        for i, (features, labels) in enumerate(train_loader):

            num_videos = features.shape[0]
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

                    if opt.apply_temporal_ot:
                        #compute q
                        p_gauss = get_cost_matrix(batch_size = opt.batch_size, num_videos = num_videos, num_videos_dataset = opt.num_videos \
                        ,sigma = opt.sigma, num_clusters = proto_scores.shape[1]) # Prior distribution, mantiene fixed order the los clusters

                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, p_gauss)
                    
                    
                    ################# AÑADIDO - PERMUTATION AWARE PRIOR ####################
                    if opt.apply_permutation_aware_prior:
                        num_frames = features.shape[0]
                        num_actions = proto_scores.shape[1]

                        """ Estimate transcripts from the frame-level pseudo-label codes Qf """
                        estimated_transcripts = estimate_transcripts(q, num_actions)

                        """ Compute Qs from T """
                        qs = compute_qs_from_transcripts(estimated_transcripts, num_actions, num_frames)

                        
                        """ Compute prior distribution Mt that imposes the permutation-aware transcript """
                        Mt = generate_matrix2(num_frames, estimated_transcripts, opt.sigma)

                        """ Compute Qa mith Mt. And use Qa for prototype loss calculation """
                        q = generate_optimal_transport_labels(proto_scores, opt.epsilon, Mt)
                    ##########################################################################

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
            
                proto_probs = F.softmax(proto_scores/opt.temperature)
                 
                if i + (epoch * len(train_loader)) % 500 == 0:
                    with torch.no_grad():
                        img = plot_confusion_matrix(proto_probs.clone().detach().cpu().numpy())
                        writer.add_image("P Matrix", img, i + (epoch * len(train_loader)))

                proto_probs = torch.clamp(proto_probs, min= 1e-30, max=1)
                proto_loss = torch.mean(torch.sum(q * torch.log(proto_probs), dim = 1))
            
            loss_tcn =  tcn_loss(embs) #>tensor(116.2414, device='cuda:0', grad_fn=<NllLossBackward0>) (va cambiando)
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

            loss_values.backward()  #  OPTIMIZA A PARTIR DE ESTA LOSS

            if i + (epoch * len(train_loader)) < opt.freeze_iters:

                for name, p in model.named_parameters():
                   
                    if "prototype" in name:
                        p.grad = None 

            
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
                    
                    logger.debug(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                         f'Cluster Loss {c_losses.val:.4f} ({c_losses.avg:.4f})\t')


        ################# AÑADIDO #############
        # After each epoch append the epoch number and the average losses (average loss across batches in the current epoch)
        with open(final_losses_path, 'a') as file:
            file.write(f'Epoch {epoch + 1}: {losses.avg:.4f}\n') # Task losses

        with open(cluster_losses_path, 'a') as file:
            file.write(f'Epoch {epoch + 1}: {c_losses.avg:.4f}\n')

        with open(TCN_losses_path, 'a') as file:
            file.write(f'Epoch {epoch + 1}: {tcn_losses.avg:.4f}\n')
        ############################

        ################# AÑADIDO #############
        """        
        # After each epoch, check if the average loss is the lowest
        if losses.avg < lowest_loss:
            lowest_loss = losses.avg
            best_epoch = epoch
            # Save the current best model state
            best_model_state = model.state_dict().copy()
        #######################################
        """


        logger.debug('loss: %f' % losses.avg)


        ############### AÑADIDO ##############
        #corpus.accuracy_corpus()
        ######################################
     
        losses.reset()
    
    f.close()

    ############# AÑADIDO ##############################
    # At the end of training, load the best model state
    #model.load_state_dict(best_model_state)
    ####################################################

    opt.resume_str = join(opt.tensorboard_dir, 'models',
                          '%s.pth.tar' % opt.log_str)
    if save:
        save_dict = {'epoch': epoch, 
                     'state_dict':  model.state_dict(), 
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



############ AÑADIDO ##################################
""" Takes the matrix q and returns a sorted list of actions by frame index """
def estimate_transcripts(q, num_actions):
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


""" Computes Qs based on the estimated transcripts """
def compute_qs_from_transcripts(transcript, num_actions, num_frames):
    # set Q_{s}^{ij} = 1 if the i-th position in T contains the j-th action
    # set Q_{s}^{ij} = 0 otherwise 

    qs = torch.zeros(num_frames, num_actions)
    for idx, action in enumerate(transcript):
        qs[idx, action] = 1
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

#########################################################

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


"""
Esta es la que había antes, por lo que devuelve la fixed order prior distribution (T del primer survey)
2D distribution, whose marginal distribution along any line perpendicular to the diagonal is a Gaussian 
distribution centered at the intersection on the diagonal
"""
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


# Just for testing
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
    sigma = 6

    Mt = generate_matrix2(num_frames, transcript, sigma)


    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(Mt, annot=False, cmap="viridis")
    plt.title("Permutation Aware Prior Distribution ")
    plt.xlabel("Action Index")
    plt.ylabel("Frame Number")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Rotate y-axis labels to 0 degrees

    plt.show()

    plt.savefig("heatmap2.png", format='png', dpi=300) 
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
    # Convert q to a NumPy array (tensor)
    q_array = np.array(q)
    q_tensor = torch.tensor(q_array, dtype=torch.float32)


    # Estimate transcripts
    transcripts = estimate_transcripts(q_tensor, num_actions)
    
    print("Sorted Action-Frame Pairs:")
    for action, frame in transcripts:
        print(f"Action {action} is most likely at frame {frame}")


"""def main():
    test1()
    test2()

if __name__ == '__main__':
    main()"""