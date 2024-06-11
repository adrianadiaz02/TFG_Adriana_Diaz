#!/usr/bin/env python

"""Transformer for relative time embedding: learn regression model in terms of
relative time.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from ute.utils.logging_setup import logger

__author__ = 'Adriana Díaz Soley'
__date__ = 'May 2024'


class TCN(nn.Module):

    def __init__(self, num_positives, temperature = 0.1):
        super(TCN, self).__init__()
        self.temperature = temperature
        self.num_positives = num_positives

        
    def _npairs_loss(self, labels, embeddings_anchor, embeddings_positive):
        """Returns n-pairs metric loss."""
        
        # Get per pair similarities.
        similarity_matrix = torch.matmul(
            embeddings_anchor, embeddings_positive.t())

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = labels.shape

        # Add the softmax loss.
        xent_loss = F.cross_entropy(
            input=similarity_matrix, target=labels)
        #xent_loss = tf.reduce_mean(xent_loss)

        return xent_loss


    def single_sequence_loss(self, embs):
        """Returns n-pairs loss for a single sequence."""

        labels = torch.arange(embs.shape[0]//self.num_positives).long().cuda()
        embs = embs/self.temperature #temperature to scale up the embeddings
        indices = torch.arange(0, embs.shape[0], step=self.num_positives) + torch.randint(0, self.num_positives - 1, size = (labels.shape[0],))
        indices_positive = indices + 1
        embeddings_anchor = embs[indices]
        embeddings_positive = embs[indices_positive]
        loss = self._npairs_loss(labels, embeddings_anchor, embeddings_positive)
        return loss

    def forward(self, embs):
        return self.single_sequence_loss(embs)



class ClusterLoss(nn.Module):

    def __init__(self, num_clusters, alpha = 0.01):
      super(ClusterLoss, self).__init__()
      self.num_clusters = num_clusters
      self.alpha = alpha

    def cluster_sim_loss(self, prototypes):

      sim = torch.matmul(prototypes, torch.transpose(prototypes, 0 , 1)) # N x N
      
      p_norm = torch.norm(prototypes, dim = 1).unsqueeze(1) # N x 1
      
      p_norm_matrix = torch.matmul(p_norm, torch.transpose(p_norm, 0, 1)) # N x N
      cosine_sim = sim / p_norm_matrix
      return cosine_sim

    def entropy_loss(self, prob, epsilon = 1e-7):
      return -torch.sum(prob*(torch.log(prob + epsilon)))


    def cosine_sim(self, embeddings, prototypes):

      sim = torch.matmul(embeddings, torch.transpose(prototypes, 0, 1))
      emb_norm = torch.norm(embeddings, dim = 1).unsqueeze(1)
      proto_norm = torch.norm(prototypes, dim  = 1).unsqueeze(1)
      norm_matrix = torch.matmul(emb_norm, torch.transpose(proto_norm, 0, 1))
      cosine_sim = sim/norm_matrix
      
      return cosine_sim

    def cluster_diversity_loss(self, batch_size, cluster_logits):

      cluster_sums = torch.sum(cluster_logits, axis = 1)
      cluster_sums = cluster_sums - batch_size/self.num_clusters
      cluster_sums = torch.max(cluster_sums, torch.tensor([0.]).cuda())
      return torch.sum(cluster_sums)

    def cluster_entropy_loss(self, probs):
    
      marginal_probs = torch.sum(probs, dim = 0).float()
      marginal_probs = torch.div(marginal_probs, probs.shape[0])
      entropy = self.entropy_loss(marginal_probs)
      return entropy
       

    def prototype_loss(self, embeddings, prototypes):

      dists = self.compute_euclidean_dist(embeddings, prototypes)
      sim = 1/dists # dist -> similarity
      sim_soft = F.softmax(sim, dim = 1) #softmax normalization
      cluster_assignments =  F.one_hot(torch.argmax(sim_soft, dim = 1), num_classes = self.num_clusters)
      cluster_assignments.requires_grad = False
      entropy = self.cluster_entropy_loss(sim_soft)
      inter_class_variances = torch.sum(cluster_assignments * (dists))
     
      return inter_class_variances, entropy, intra_class_variances

    def forward(self, embeddings, prototypes):
      return self.prototype_loss(embeddings, prototypes)


    def compute_euclidean_dist(self,  embeddings, prototypes):
      dists = torch.sum(embeddings**2, dim = 1).view(-1, 1) + torch.sum(prototypes**2, dim = 1).view(1, -1) -2 * torch.matmul(embeddings, torch.transpose(prototypes, 0, 1)) 
      #print(dists.shape)
      return dists

#####################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, opt, num_clusters, dim_feedforward=2048):
        super(TransformerEncoder, self).__init__()
        self.opt = opt
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(opt.feature_dim, opt.transformer_dropout)

        # Reduce dimension from feature_dim to embed_dim
        self.dim_reduction = nn.Linear(opt.feature_dim, opt.embed_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model = opt.embed_dim, 
                                                    nhead = opt.transformer_num_heads,
                                                    dim_feedforward = dim_feedforward, 
                                                    dropout = opt.transformer_dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=opt.transformer_num_layers)

        # Prototype layer 
        self.prototype_layer = nn.Linear(opt.embed_dim, num_clusters, bias=False)


        if opt.time_loss:
          self.fc_last = nn.Linear(opt.embed_dim, 1)

        ### PARA EL EMBEDDED 
        self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        self.fc2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim)

        # Initialize weights
        self._init_weights()


    def forward(self, x):
        """
        obs:  forward pass of the model, including calculations up to the output, maintains numerical 
        stability with no NaN or Inf values detected (print("Input check:", torch.isfinite(x).all()), etc)
        """
        # obs: inputs con batch size 512 y num_videos 1 --> number of frames = 512
        # feature dimension = 64; embedding dimension = 40
        #print("input: ", x.shape) # >> [512, 1, 64]
        
        x = self.pos_encoder(x)
        #print("after positional encoder: ", x.shape) # >> [512, 1, 64]
        
        x = self.dim_reduction(x)  # Reduce dimension before feeding into Transformer
        #print("after dimensionality reduction: ", x.shape) # >> [512, 1, 40]

        x = self.transformer_encoder(x)
        #print("after transformer encoder: ", x.shape) # >> [512, 1, 40]

        # Flatten the batch and sequence dimensions
        batch_size, seq_len, feature_dim = x.shape
        x = x.view(batch_size * seq_len, feature_dim)  # Flatten to [batch_size * seq_len, feature_dim]

        # Get prototype scores and embeddings before potential final transformation
        proto_scores = self.prototype_layer(x)
        #print("proto_scores: ", proto_scores.shape) # >> [512, 7]

        embs = x.clone()  # Copy x to embs before any further manipulation

        if self.opt.time_loss:
            x = self.fc_last(x)  # Apply final linear transformation

        return x, proto_scores, embs
    
    def embedded(self, x):
        shape_dims = [*x.shape]
        x = self.fc1(x)
        x = F.relu(x) # CAMBIADO A RELU, antes sigmoid
        x = self.fc2(x)
        x = F.relu(x) # CAMBIADO A RELU, antes sigmoid
        return x

    def get_prototypes(self):
        return self.prototype_layer.weight.data.clone()
    
    # He initialization (also known as Kaiming initialization)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def print_initial_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.data}")

    def update_prototypes(self, new_prototypes):
      with torch.no_grad():
        self.prototype_layer.weight.copy_(torch.tensor(new_prototypes).to(self.prototype_layer.weight.device))
     
##############################################################


def create_model(opt, num_clusters, learn_prototype = True):

    torch.manual_seed(opt.seed)
    
    print("Creating Transformer")

    model = TransformerEncoder(num_clusters = num_clusters, opt = opt).to(opt.device)
    
    loss = nn.MSELoss(reduction='sum')
    tcn_loss = TCN(int((opt.batch_size/opt.num_videos)/opt.num_splits))
    print(int((opt.batch_size/opt.num_videos)/opt.num_splits))
    
    # loss = nn.MSELoss().cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params,
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))

    return model, loss, tcn_loss, optimizer

