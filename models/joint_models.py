import os, sys, pickle
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from itertools import combinations, permutations

from utils import *
from layers import *
from functions import *
from data import *

from .base import *

import copy

# torch.backends.cudnn.enabled = False


class TabEncoding(nn.Module):
    
    def __init__(self, config, mdrnn, emb_dim=None):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim  
        self.extra_hidden_dim = config.head_emb_dim
        
        self.seq2mat = CatReduce(n_in=self.emb_dim, n_out=self.hidden_dim)
        if self.extra_hidden_dim > 0:
            self.reduce_X = nn.Linear(self.emb_dim+self.extra_hidden_dim, self.hidden_dim)
        
        self.mdrnn = mdrnn
        
    def forward(
        self, 
        S: torch.Tensor, 
        Tlm: torch.Tensor = None,
        Tstates: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        
        if masks is not None:
            masks = masks.unsqueeze(1) & masks.unsqueeze(2)
        
        X = self.seq2mat(S, S) # (B, N, N, H)
        X = F.relu(X)
        if self.extra_hidden_dim > 0:
            X = self.reduce_X(torch.cat([X, Tlm], -1))
            X = F.relu(X)
            
        T, Tstates = self.mdrnn(X, states=Tstates, masks=masks)
        
        # the content of T and Tstates is the same;
        # keep Tstates is to save some slice and concat operations
        
        return T, Tstates
    
    
class SeqEncoding(nn.Module):
    
    def __init__(self, config, emb_dim=None, n_heads=8):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim  
        self.n_heads = n_heads
        
        self.linear_qk = nn.Linear(self.emb_dim, self.n_heads, bias=None)
        self.linear_v = nn.Linear(self.emb_dim, self.hidden_dim, bias=True)
        self.linear_o = nn.Linear(self.emb_dim, self.hidden_dim, bias=True)
        
        self.norm0 = nn.LayerNorm(self.config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim*4, self.config.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(self.config.hidden_dim)
        
        self.dropout_layer = nn.Dropout(self.config.dropout)
        
    def forward(
        self, 
        S: torch.Tensor,
        T: torch.Tensor,
        masks: torch.Tensor = None,
    ):
        B, N, H = S.shape
        n_heads = self.n_heads
        subH = H // n_heads
        
        if masks is not None:
            masks = masks.unsqueeze(1) & masks.unsqueeze(2)
            masks_addictive = -1000. * (1. - masks.float())
            masks_addictive = masks_addictive.unsqueeze(-1)
            
        S_res = S
        
        # Table-Guided Attention
        attn = self.linear_qk(T)
        attn = attn + masks_addictive
        attn = attn.permute(0, -1, 1, 2) # (B, n_heads, N, T)
        attn = attn.softmax(-1) # (B, n_heads, N, T)
        
        v = self.linear_v(S) # (B, N, H)
        v = v.view(B, N, n_heads, subH).permute(0, 2, 1, 3) # B, n_heads, N, subH
        
        S = attn.matmul(v) # B, n_heads, N, subH
        S = S.permute(0, 2, 1, 3).reshape(B, N, subH*n_heads)
        
        S = self.linear_o(S)
        S = F.relu(S)
        S = self.dropout_layer(S)
        
        S = S_res + S
        S = self.norm0(S)
        
        S_res = S
        
        # Position-wise FeedForward
        S = self.ffn(S)
        S = self.dropout_layer(S)
        S = S_res + S
        S = self.norm1(S)
        
        return S, attn
    
    
class TwoWayEncoding(nn.Module):
    
    def __init__(self, config, emb_dim=None, n_heads=8, first=False, direction='B'):
        super().__init__()
        
        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim  
        self.n_heads = n_heads
        self.extra_hidden_dim = config.head_emb_dim  
        self.first = first # first MD-RNN only needs 2D, others should be 3D
        self.direction = direction
        
        self.tab_encoding = TabEncoding(
            config=self.config, 
            mdrnn=get_mdrnn_layer(
                config, direction=direction, norm='', Md='25d', first_layer=first), 
            emb_dim=self.emb_dim,)
        self.seq_encoding = SeqEncoding(
            config=self.config, emb_dim=self.emb_dim, n_heads=n_heads)
        
    def forward(
        self, 
        S: torch.Tensor, 
        Tlm: torch.Tensor = None,  # T^{\ell} attention weights of BERT
        Tstates: torch.Tensor = None, 
        masks: torch.Tensor = None
    ):
        
        T, Tstates = self.tab_encoding(S=S.clone(), Tlm=Tlm, Tstates=Tstates, masks=masks)
        
        S, attn = self.seq_encoding(S=S, T=T.clone(), masks=masks)
        
        return S, attn, T, Tstates
    
    
class StackedTwoWayEncoding(nn.Module):
    
    def __init__(self, config, n_heads=8, direction='B'):
        super().__init__()
        
        self.config = config
        self.depth = config.num_layers
        self.hidden_dim = config.hidden_dim  
        self.extra_hidden_dim = config.head_emb_dim
        self.n_heads = n_heads
        self.direction = direction
        
        self.layer0 = TwoWayEncoding(
            self.config, n_heads=n_heads, first=True, direction=direction,
        )
        
        self.layers = nn.ModuleList([
            TwoWayEncoding(
                self.config, n_heads=n_heads, first=False, direction=direction,
            ) for i in range(self.depth-1)
        ])
        
    def forward(
        self, 
        S: torch.Tensor, 
        Tlm: torch.Tensor = None, 
        masks: torch.Tensor = None
    ):
        S_list = []
        T_list = []
        attn_list = []
        
        Tstates = None
        
        S, attn, T, Tstates = self.layer0(S=S, Tlm=Tlm, Tstates=Tstates, masks=masks)
            
        S_list.append(S)
        T_list.append(T)
        attn_list.append(attn) 
        
        for layer in self.layers:
            
            S, attn, T, Tstates = layer(S=S, Tlm=Tlm, Tstates=Tstates, masks=masks)
            
            S_list.append(S)
            T_list.append(T)
            attn_list.append(attn) 
        
        return S_list, T_list, attn_list


class ConvNet3layers(nn.Module):

    def __init__(self, input_dim, hid_dim, output_dim, kernel_size=1, stride=1, padding=0, dropout=0.3):
        super(ConvNet3layers, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hid_dim, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(hid_dim, output_dim, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        return self.conv3(x)


class JointModel(Tagger):
    
    def check_attrs(self):
        assert hasattr(self, 'ner_tag_indexing')
        assert hasattr(self, 're_tag_indexing')
    
    def get_default_trainer_class(self):
        return JointTrainer
    
    def set_embedding_layer(self):
        
        if self.config.tag_form.lower() == 'iob2':
            self.one_entity_n_tags = 2
        elif self.config.tag_form.lower() == 'iobes':
            self.one_entity_n_tags = 4
        else:
            raise Exception('no such tag form.')
        self.ner_tag_indexing = get_tag_indexing(self.config)
        self.re_tag_indexing = IndexingMatrix(depth=3) # 2d
        self.re_tag_indexing.vocab = {'O': 0}
        
        self.token_embedding = AllEmbedding(self.config)
        self.token_indexing = self.token_embedding.preprocess_sentences

        # Trung
        self.conv_encoder = ConvNet3layers(input_dim=self.config.hidden_dim,
                                           hid_dim=self.config.hidden_dim//2, output_dim=1)
        self.config.max_sent_length = 120  # Conll04
        # self.config.size_embedding_size = 100
        # self.size_embeddings = nn.Embedding(self.config.max_sent_length, self.config.size_embedding_size)
        
    def set_encoding_layer(self):
        
        emb_dim = self.config.token_emb_dim + self.config.char_emb_dim + self.config.lm_emb_dim
        
        self.reduce_emb_dim = nn.Linear(emb_dim, self.config.hidden_dim)
        init_linear(self.reduce_emb_dim)
        
        self.bi_encoding = StackedTwoWayEncoding(self.config)
        
        self.dropout_layer = nn.Dropout(self.config.dropout)
        
    def set_logits_layer(self):
        # Trung: entry layer
        # (batch_size, num_rows, num_cols, 1) -> (batch_size)
        self.entry_tag_logits_layer = nn.Linear(self.config.max_sent_length ** 2, 1)
        
    def set_loss_layer(self):
        
        if self.config.crf:
            self.crf_layer = eval(self.config.crf)(self.config)
        
        self.soft_loss_layer = LabelSmoothLoss(0.02, reduction='none') 
        self.loss_layer = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss_layer = nn.MSELoss()
    
    def forward_embeddings(self, inputs):
        
        if '_tokens' in inputs:
            sents = inputs['_tokens']
        else:
            sents = inputs['tokens']

        # Sentence length
        inputs['entry_length'] = torch.FloatTensor(inputs['entry_length']).to(self.device)
        inputs['sent_length'] = torch.LongTensor(inputs['sent_length']).to(self.device)
        # sent_length_embeddings = self.size_embeddings(inputs['sent_length'])
            
        embeddings, masks, embeddings_dict = self.token_embedding(sents, return_dict=True)
        
        embeddings = self.dropout_layer(embeddings)
        embeddings = self.reduce_emb_dim(embeddings)
            
        lm_heads = embeddings_dict.get('lm_heads', None)
        inputs['lm_heads'] = lm_heads
        
        ## relation encoding
        seq_embeddings, tab_embeddings, attns = self.bi_encoding(embeddings, masks=masks, Tlm=lm_heads)
        seq_embeddings = seq_embeddings[-1]
        tab_embeddings = tab_embeddings[-1]
        seq_embeddings = self.dropout_layer(seq_embeddings)
        tab_embeddings = self.dropout_layer(tab_embeddings)
        
        inputs['attns'] = attns
        inputs['masks'] = masks
        inputs['tab_embeddings'] = tab_embeddings
        inputs['seq_embeddings'] = seq_embeddings
        # inputs['sent_length_embeddings'] = sent_length_embeddings

        return inputs
    
    
    def forward_step(self, inputs):
        '''
        inputs: {
            'tokens': List(List(str)),
            '_tokens'(*): [Tensor, Tensor],
        }
        outputs: +{
            'logits': Tensor,
            'masks': Tensor
        }
        '''
        inputs = self.forward_embeddings(inputs)
        tab_embeddings = inputs['tab_embeddings']
        seq_embeddings = inputs['seq_embeddings']
        # sent_length_embeddings = inputs['sent_length_embeddings']

        # Use conv net with 1x1 kernel
        tab_embeddings = tab_embeddings.permute(0, 3, 1, 2)

        # Trung: transform seq embeddings into a mask then apply a linear layer on it
        batch_size = tab_embeddings.shape[0]

        entry_tag_logits = self.conv_encoder(tab_embeddings)
        # entry_tag_logits = F.relu(entry_tag_logits)
        entry_tag_logits = entry_tag_logits.view(batch_size, -1)

        pad_amount = self.config.max_sent_length ** 2 - inputs['batch_max_length'] ** 2
        entry_tag_logits = F.pad(input=entry_tag_logits, pad=[0, pad_amount], mode='constant', value=0)
        entry_tag_logits = self.entry_tag_logits_layer(entry_tag_logits).view(batch_size,)
        # print('post.entry_tag_logits: ', entry_tag_logits)
        # print('loss value: ', self.mse_loss_layer(entry_tag_logits, inputs['entry_length']).item())
        # exit()
        
        rets = inputs
        rets['entry_tag_logits'] = entry_tag_logits
        
        return rets

    def forward(self, inputs):

        rets = self.forward_step(inputs)
        entry_tag_logits = rets['entry_tag_logits']
        entry_loss = self.mse_loss_layer(entry_tag_logits, inputs['entry_length'])
        
        loss = entry_loss
        rets['loss'] = loss
            
        return rets

    def predict_step(self, inputs):
        
        rets = self.forward_step(inputs)
        entry_tag_logits = rets['entry_tag_logits']

        rets['entry_length_preds'] = entry_tag_logits
            
        return rets
    
    def train_step(self, inputs):
        
        self.need_update = True
        
        rets = self(inputs)
        loss = rets['loss']
        loss.backward()
        
        grad_period = self.config.grad_period if hasattr(self.config, 'grad_period') else 1
        if grad_period == 1 or self.global_steps.data % grad_period == grad_period-1:
            nn.utils.clip_grad_norm_(self.parameters(), 5)
        
        return rets
    
    def save_ckpt(self, path):
        torch.save(self.state_dict(), path+'.pt')
        with open(path+'.vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.token_indexing.vocab, f)
        with open(path+'.char_vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.char_indexing.vocab, f)
        with open(path+'.ner_tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.ner_tag_indexing.vocab, f)
        with open(path+'.re_tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.re_tag_indexing.vocab, f)
            
    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path+'.pt'))
        with open(path+'.vocab.pkl', 'rb') as f:
            self.token_embedding.token_indexing.vocab = pickle.load(f)
            self.token_embedding.token_indexing.update_inv_vocab()
        with open(path+'.char_vocab.pkl', 'rb') as f:
            self.token_embedding.char_indexing.vocab = pickle.load(f)
            self.token_embedding.char_indexing.update_inv_vocab()
        with open(path+'.ner_tag_vocab.pkl', 'rb') as f:
            self.ner_tag_indexing.vocab = pickle.load(f)
            self.ner_tag_indexing.update_inv_vocab()
        with open(path+'.re_tag_vocab.pkl', 'rb') as f:
            self.re_tag_indexing.vocab = pickle.load(f)
            self.re_tag_indexing.update_inv_vocab()

    
    
class JointModelMacroF1(JointModel):
    
    def get_default_trainer_class(self):
        return JointTrainerMacroF1
    