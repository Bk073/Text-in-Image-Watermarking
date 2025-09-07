import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, vocab
from torchtext.utils import download_from_url, extract_archive
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

import pdb

import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from torchvision import datasets as dv
from torchtext import datasets as dt
from torchtext.vocab import vocab, GloVe
from torchtext.data.utils import get_tokenizer

from torchtext.data.metrics import bleu_score

import numpy as np

from torchinfo import summary
import string
import os
import glob
import random
from matplotlib import pyplot as plt
from PIL import Image

from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torchinfo import summary
import re
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import random
from skimage.util import random_noise
import io
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger



def download_data():
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.de.gz', 'train.en.gz')
    val_urls = ('val.de.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

    return train_filepaths, val_filepaths, test_filepaths

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\+d', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(preprocess(string_)))
  
    v = vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], special_first=True)
    v.set_default_index(0)
    return v

def data_process(filepaths, en_vocab, en_tokenizer):
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for raw_en in raw_en_iter:
        en_token = [en_vocab[token] for token in en_tokenizer(preprocess(raw_en))][:13]
        if len(en_token) != 13:
            en_token.extend((13-len(en_token))*[1])
        en_tensor_ = torch.tensor(en_token,
                                dtype=torch.long)
        data.append(en_tensor_)
    return data

def remove_image(path_):
    remove_list = []
    for i in glob.glob(path_+'*/*'):
        img = np.array(Image.open(os.path.join(path_,i)))
        if img.shape[-1] != 3:
            remove_list.append(os.path.join(path_,i))
    return remove_list

def random_noise(x, min_value, max_value):
    noise = torch.rand(x.shape) * (max_value - min_value) + min_value
    return noise.type_as(x)


class WatermarkDataset(Dataset):
  def __init__(self, cover_img_dir, remove_list, watermark_text, vocab, transform_c=None, data_size=None):
    self.tokens = []
    self.cover_img_dir = cover_img_dir
    self.ls = []
    self.transform_c = transform_c
    PAD_IDX = vocab['<pad>']
    BOS_IDX = vocab['<sos>']
    EOS_IDX = vocab['<eos>']
    
    for en_item in watermark_text:
        self.tokens.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  
    self.tokens = pad_sequence(self.tokens, padding_value=1)
    
#     self.input_ids = torch.tensor(self.tokens, dtype=torch.long)
    for i in os.listdir(self.cover_img_dir):
            if i.split('.')[-1] == 'JPEG':
                self.ls.append(os.path.join(self.cover_img_dir,i))
    for i in glob.glob(self.cover_img_dir+'*/*'):
          self.ls.append(i)
                
    self.remove_list = remove_list
    self.ls = list(set(self.ls) - set(self.remove_list))
    
    random.shuffle(self.ls)
    
    self.cover_image_list = self.ls[:data_size]
    self.input_ids = self.tokens[:data_size]

  def __len__(self):
    return len(self.cover_image_list)
  
  def __getitem__(self, index):
    input_ids = self.input_ids[:, index]
    cover_img_pt = self.cover_image_list[index]

    cover_img = Image.open(cover_img_pt)
    cover_img = T.ToTensor()(cover_img)

    if self.transform_c:
        cover_img = self.transform_c(cover_img)
        cover_img = T.ToTensor()(cover_img)
    return cover_img, input_ids


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

#         self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True) #changed
        self.rnn = nn.GRU(emb_dim, enc_hid_dim)

#         self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim) #changed
        self.layer_norm_h = nn.LayerNorm(self.emb_dim)
        self.layer_norm_out = nn.LayerNorm(self.emb_dim)
        self.layer_norm_1 = nn.LayerNorm(self.emb_dim)
        self.layer_norm_2 = nn.LayerNorm(self.emb_dim)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.fc_2 = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:
#         print(src.shape)
        embedded = self.dropout(self.embedding(src))
#         print("Source shape: ", src.shape)
        outputs, hidden = self.rnn(embedded)
#         print("H1 shape: ", hidden.shape)
#         hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) #changed
        hidden = self.layer_norm_h(hidden)
        outputs = self.layer_norm_out(outputs)
        hidden = self.fc(hidden)
        hidden = self.layer_norm_1(hidden)
#         hidden = torch.tanh(hidden)
        outputs = self.fc_2(outputs)
        outputs = self.layer_norm_2(outputs)
#         outputs = torch.tanh(outputs)
        outputs = torch.cat((outputs, hidden), dim=0)
        return outputs


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

#         self.attn_in = (enc_hid_dim * 2) + dec_hid_dim changed
        self.attn_in = enc_hid_dim + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]
#         pdb.set_trace()
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)
#         pdb.set_trace()
        return F.softmax(attention, dim=1)


def random_noise(x, min_value, max_value):
    noise = torch.rand(x.shape) * (max_value - min_value) + min_value
    return noise.type_as(x)

class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
#         self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim) changed
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim)
        self.layer_norm_h = nn.LayerNorm(self.emb_dim)
        self.layer_norm_o = nn.LayerNorm(self.emb_dim)
        self.layer_norm_r = nn.LayerNorm(self.emb_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
#         pdb.set_trace()
        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:
        
        input = input.unsqueeze(0) # 128
        # hidden -> 128, 64
        # de
        embedded = self.dropout(self.embedding(input))
        
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)
#         pdb.set_trace()
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        output = self.layer_norm_o(output)
        decoder_hidden = self.layer_norm_h(decoder_hidden)
        weighted_encoder_rep = self.layer_norm_r(weighted_encoder_rep)

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Block_inc(nn.Module):
    def __init__(self, in_channel = 3, out_channel=8,decoder=False, **kwargs):
        super().__init__()
        self.decoder = decoder
        self.conv_1 = nn.Conv2d(in_channel, out_channel, (3,3), 1, padding='same')
        self.bn_1 = nn.BatchNorm2d(out_channel)

        self.conv_2 = nn.Conv2d(out_channel, out_channel, (3,3), 1 , padding='same')
        self.bn_2 = nn.BatchNorm2d(out_channel)
        
        self.conv_3 = nn.Conv2d(out_channel, out_channel, (3,3), 1, padding='same')
        self.bn_3 = nn.BatchNorm2d(out_channel)

        self.conv_4 = nn.Conv2d(out_channel, out_channel, (3,3), 1, padding='same')
        self.bn_4 = nn.BatchNorm2d(out_channel)
        
        if self.decoder:
            self.conv_5 = nn.Conv2d(out_channel, out_channel, (3,3), 1, padding='same')
            self.bn_5 = nn.BatchNorm2d(out_channel)

            self.conv_6 = nn.Conv2d(out_channel, out_channel, (3,3), 1, padding='same')
            self.bn_6 = nn.BatchNorm2d(out_channel)

            self.conv_7 = nn.Conv2d(out_channel, out_channel, (3,3), 1, padding='same')
            self.bn_7 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()
        self.apply(self._weights_init)
  
    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
    
    def forward(self, input, decoder=False):
        c_1 = self.conv_1(input)
        b_1 = self.bn_1(c_1)
        b_1 = self.relu(b_1)
        
        c_2 = self.conv_2(b_1)
        b_2 = self.bn_2(c_2)
        b_2 = self.relu(b_2)

        c_3 = self.conv_3(b_1)
        b_3 = self.bn_3(c_3)
        b_3 = self.relu(b_3)
        
        c_4 = self.conv_4(b_3)
        b_4 = self.bn_4(c_4)
        out = self.relu(b_4)
        
        if self.decoder:
            b_4 = self.relu(out)
            
            c_5 = self.conv_5(b_4)
            b_5 = self.bn_5(c_5)
            b_5 = self.relu(b_5)
            
            c_6 = self.conv_6(b_5)
            b_6 = self.bn_6(c_6)
            b_6 = self.relu(b_6)
            
            c_7 = self.conv_7(b_6)
            b_7 = self.bn_7(c_7)
            out = self.relu(b_7)
        
        return out


class Embedder64(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block_inc = Block_inc(3, 64)
        self.block_inc_2 = Block_inc(68, 68)

        self.conv1_wm = nn.Conv2d(68, 70, (3,3), 1, padding='same')
        self.bn_1 = nn.BatchNorm2d(70)
        
#         self.conv2_wm = nn.Conv2d(70, 3, (3,3), 1, padding='same') 
        ''' There is some problem and the watermark image is not of good quality on rotation, so I thought of adding
        more convolutional layers. The other thing I noticed in Embedder was decreasing 70 channel to 3, lets play with it 
        and add more convolutional layers and decrease the channel gradually.'''
#         self.conv2_wm = nn.Conv2d(70, 32, (3,3), 1, padding='same') changed size
        self.conv2_wm = nn.Conv2d(73, 32, (3,3), 1, padding='same')
        self.bn_2 = nn.BatchNorm2d(32)
        
        self.conv3_wm = nn.Conv2d(32, 16, (3, 3), 1, padding='same')
        self.bn_3 = nn.BatchNorm2d(16)
        
#         self.conv4_wm = nn.Conv2d(16, 8, (3, 3), 1, padding='same')
        self.conv4_wm = nn.Conv2d(20, 8, (3, 3), 1, padding='same')
        self.bn_4 = nn.BatchNorm2d(8)
        
        self.conv5_wm = nn.Conv2d(9, 3, (3, 3), 1, padding='same')
        self.bn_5 = nn.BatchNorm2d(3)
        
#         self.skip_1_conv = nn.Conv2d(64, 70, (3, 3), 1, padding='same')
#         self.skip_1_bn = nn.BatchNorm2d(70)
        
#         self.skip_2_conv = nn.Conv2d(3, 16, (3, 3), 1, padding='same')
#         self.skip_2_bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, cover_img, watermark):
        batch = watermark.shape[0]
        watermark = watermark.permute(1, 0, 2)
        w = watermark.repeat(1, 64, 1)
#         w = watermark.repeat(1, 128, 1)
#         pdb.set_trace()
#         print(w.shape)
        # try:
        w = torch.reshape(w, (watermark.shape[0], -1, 256, 256)) # watermark -> 15, 17, 64, w -> 15, 1088, 64
        # except:
            # pdb.set_trace()

        c_img = self.block_inc(cover_img)
        # print(c_img.shape, w.shape, cover_img.shape)
        # try:
        watermark = torch.cat((c_img, w, cover_img), dim=1) # 1+3+64 = 68
        # except:
            # pdb.set_trace()
        watermark = self.block_inc_2(watermark)
        
        watermark = self.conv1_wm(watermark)
        watermark = self.bn_1(watermark)
            # skip connection
#         c_img_1 = self.skip_1_conv(c_img)
#         c_img_1 = self.skip_1_bn(c_img_1)
# #         c_img = self.relu(c_img)

#         watermark = c_img_1 + watermark
#         watermark = self.relu(watermark)
        watermark = torch.cat([watermark, cover_img], 1)
        
        watermark = self.conv2_wm(watermark)
        watermark = self.bn_2(watermark)
        watermark = self.relu(watermark)
        
        watermark = self.conv3_wm(watermark)
        watermark = self.bn_3(watermark)
        # skip connection
#         c_img_2 = self.skip_2_conv(cover_img)
#         c_img_2 = self.skip_2_bn(c_img_2)
#         watermark = c_img_2 + watermark
# #         print(watermark.shape)
#         watermark = self.relu(watermark)
        watermark = torch.cat([watermark, cover_img, w], 1)
        
        watermark = self.conv4_wm(watermark)
        watermark = self.bn_4(watermark)
        watermark = self.relu(watermark)
        
        watermark = torch.cat([watermark, w], 1)
        watermark = self.conv5_wm(watermark)
        watermark = self.bn_5(watermark)
#         watermark = self.relu(watermark)
        
        watermark = self.sigmoid(watermark)
        return watermark

# class Extractor64(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.block_inc_1 = Block_inc()
#         self.block_inc_2 = Block_inc(64, 64)
# #         self.avg_pool = nn.AvgPool2d(3,3)
# #         self.avg_pool_2 = nn.AvgPool2d(2,2)
        
#         self.conv_1 = nn.Conv2d(8, 16, 3, 2)
#         self.bn_1 = nn.BatchNorm2d(16)
        
#         self.conv_2 = nn.Conv2d(16, 32, 3, 3,)
#         self.bn_2 = nn.BatchNorm2d(32)
        
# #         self.conv_3 = nn.Conv2d(2, 1, 5)
#         self.conv_3 = nn.Conv2d(32, 64, 3, 3)
#         self.bn_3 = nn.BatchNorm2d(64)
        
#         self.conv_4 = nn.Conv2d(64, 64, 3, 3)
#         self.bn_4 = nn.BatchNorm2d(64)
        
# #         self.conv_5 = nn.Conv2d(64, 256, 2, 2)
#         self.conv_5 = nn.Conv2d(64, 128, 2, 1)
#         self.bn_5 = nn.BatchNorm2d(128)
        
#         self.conv_6 = nn.Conv2d(128,256, 2, 1)
#         self.bn_6 = nn.BatchNorm2d(256)
        
#         self.conv_7 = nn.Conv2d(256, 512, 2, 1)
#         self.bn_7 = nn.BatchNorm2d(512)
        
# #         self.bn_b_1 = nn.BatchNorm2d(6)
# #         self.bn_b_2 = nn.BatchNorm2d(6)
        
# #         self.conv_4 = nn.Conv2d(3, 3, 2, 2)
# #         self.linear_1 = nn.Linear(768, 2700)
# #         self.dropout = nn.Dropout(p=0.1) # 0.5

#         self.relu = nn.ReLU()
# #         self.tanh = 

#     def forward(self, watermark_img): 
#         watermark = self.block_inc_1(watermark_img) #1. 64x256x256
# #         watermark = self.avg_pool(watermark)
    
#         watermark = self.conv_1(watermark) # 80x105x105
#         watermark = self.bn_1(watermark)
#         watermark = self.relu(watermark) # 50
# #         watermark = self.avg_pool(watermark)
# #         watermark = self.dropout(watermark)
        
# #         print(watermark.shape)
#         watermark = self.conv_2(watermark) # 100x52x52
#         watermark = self.bn_2(watermark)
#         watermark = self.relu(watermark) # 25
        
#         watermark = self.conv_3(watermark)
#         watermark = self.bn_3(watermark)
#         watermark = self.relu(watermark)
# #         watermark = self.dropout(watermark)
        
#         watermark = self.conv_4(watermark)
#         watermark = self.bn_4(watermark)
#         watermark = self.relu(watermark)
# #         print(watermark.shape)
#         watermark = self.block_inc_2(watermark)
        
#         watermark = self.conv_5(watermark)
#         watermark = self.bn_5(watermark)
#         watermark = self.relu(watermark)
        
#         watermark = self.conv_6(watermark)
#         watermark = self.bn_6(watermark)
#         watermark = self.relu(watermark)
        
#         watermark = self.conv_7(watermark)
#         watermark = self.bn_7(watermark)
#         watermark = torch.tanh(watermark) # added this to make the range same as the encoder
# #         print(watermark.shape)
# #         print(watermark.mean([2, 3]).shape)
#         watermark = torch.reshape(watermark, (watermark.size()[0], 8, 64)) # 16, batch, 64
#         return watermark.permute(1, 0, 2)

class Extractor64(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_inc_1 = Block_inc()
        self.block_inc_2 = Block_inc(64, 64)
#         self.avg_pool = nn.AvgPool2d(3,3)
#         self.avg_pool_2 = nn.AvgPool2d(2,2)
        
        self.conv_1 = nn.Conv2d(8, 16, 3, 2)
        self.bn_1 = nn.BatchNorm2d(16)
        
        self.conv_2 = nn.Conv2d(16, 32, 3, 3,)
        self.bn_2 = nn.BatchNorm2d(32)
        
#         self.conv_3 = nn.Conv2d(2, 1, 5)
        self.conv_3 = nn.Conv2d(32, 64, 3, 3)
        self.bn_3 = nn.BatchNorm2d(64)
        
        self.conv_4 = nn.Conv2d(64, 64, 3, 3)
        self.bn_4 = nn.BatchNorm2d(64)
        
#         self.conv_5 = nn.Conv2d(64, 256, 2, 2)
        self.conv_5 = nn.Conv2d(64, 128, 2, 1)
#         self.conv_5 = nn.Conv2d(64, 128, 1, 1)
        self.bn_5 = nn.BatchNorm2d(128)
        
        self.conv_6 = nn.Conv2d(128,256, 2, 1)
#         self.conv_6 = nn.Conv2d(128, 256, 1, 1)
        self.bn_6 = nn.BatchNorm2d(256)
        
        self.conv_7 = nn.Conv2d(256, 512, 2, 1)
        self.bn_7 = nn.BatchNorm2d(512)
        
        self.conv_8 = nn.Conv2d(512, 1024, 1)
        self.bn_8 = nn.BatchNorm2d(1024)
        
#         self.bn_b_1 = nn.BatchNorm2d(6)
#         self.bn_b_2 = nn.BatchNorm2d(6)
        
#         self.conv_4 = nn.Conv2d(3, 3, 2, 2)
#         self.linear_1 = nn.Linear(768, 2700)
#         self.dropout = nn.Dropout(p=0.1) # 0.5

#         self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
#         self.tanh = 

    def forward(self, watermark_img): 
        watermark = self.block_inc_1(watermark_img) #1. 64x256x256

#         watermark = self.avg_pool(watermark)
    
        watermark = self.conv_1(watermark) # 80x105x105
        watermark = self.bn_1(watermark)
        watermark = self.relu(watermark) # 50
#         watermark = self.avg_pool(watermark)
#         watermark = self.dropout(watermark)
        
#         print(watermark.shape)
        watermark = self.conv_2(watermark) # 100x52x52
        watermark = self.bn_2(watermark)
        watermark = self.relu(watermark) # 25
        
        watermark = self.conv_3(watermark)
        watermark = self.bn_3(watermark)
        watermark = self.relu(watermark)
#         watermark = self.dropout(watermark)
        
        watermark = self.conv_4(watermark)
        watermark = self.bn_4(watermark)
        watermark = self.relu(watermark)
#         print(watermark.shape)
        watermark = self.block_inc_2(watermark)
        
        watermark = self.conv_5(watermark)
        watermark = self.bn_5(watermark)
        watermark = self.relu(watermark)
        
        watermark = self.conv_6(watermark)
        watermark = self.bn_6(watermark)
        watermark = self.relu(watermark)
        
        watermark = self.conv_7(watermark)
        watermark = self.bn_7(watermark)
        watermark = self.relu(watermark)
        
        watermark = self.conv_8(watermark)
        watermark = self.bn_8(watermark)
#         watermark = torch.tanh(watermark) # added this to make the range same as the encoder
#         print(watermark.shape)
#         print(watermark.mean([2, 3]).shape)
        watermark = torch.reshape(watermark, (watermark.size()[0], 16, 64)) # 16, batch, 64
#         watermark = torch.reshape(watermark, (watermark.size()[0], 8, 64)) # 16, batch, 64
        
        return watermark.permute(1, 0, 2)


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        # 3 blocks, 64 channels
        layers = [ConvBNRelu(3, 64)]
        for _ in range(3):
            layers.append(ConvBNRelu(64, 64))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(64, 1)

    def forward(self, image):
        X = self.before_linear(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X


class EndtoEnd(pl.LightningModule):
    def __init__(self, encoder, decoder, embedder, extractor, discriminator, teacher_forcing=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing
        self.discriminator = discriminator
        self.embedder = embedder
        self.extractor = extractor
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.ssim = SSIM()
        self.train_loss = []

    def training_step(self, batch, batch_idx,  optimizer_idx):
        # sen = batch['text_data']
        # img = batch['image_data']
        sen = batch[1].permute(1, 0)
        img = batch[0]
        batch_size = sen.shape[1]
        max_len = sen.shape[0]
        trg_vocab_size = self.decoder.output_dim
        cover_label = 1
        encoded_label = 0
        train_loss = []
        embedding_loss = []
        image_loss = []
        text_loss = []
        ssim_total = []
        adversarial_bce = []
        discr_cover_bce = []
        discr_encod_bce = []
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).float()
        outputs = outputs.type_as(sen)
        outputs = outputs.type(torch.float32)
        d_target_label_cover = torch.full((batch_size, 1), cover_label, dtype=torch.float32)
        d_target_label_cover = d_target_label_cover.type_as(sen)
        d_target_label_cover = d_target_label_cover.type(torch.float32)
        d_target_label_encoded = torch.full((batch_size, 1), encoded_label, dtype=torch.float32)
        d_target_label_encoded = d_target_label_encoded.type_as(sen)
        d_target_label_encoded = d_target_label_encoded.type(torch.float32)
        g_target_label_encoded = torch.full((batch_size, 1), cover_label, dtype=torch.float32)
        g_target_label_encoded = g_target_label_encoded.type_as(sen)
        g_target_label_encoded = g_target_label_encoded.type(torch.float32)

        if optimizer_idx == 1:
            d_on_cover = self.discriminator(img)
            # pdb.set_trace()
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            self.log("dis_loss_real_train", d_loss_on_cover)
            # print("Cover"*5)
            return d_loss_on_cover
        else:
            encoder_outputs = self.encoder(sen)
            # print(img.shape, print(" main"*5))
            # if img.shape[0] != 16:
            #     pdb.set_trace()
            watermark_img = self.embedder(img, encoder_outputs)
            # train on fake
            d_on_encoded = self.discriminator(watermark_img)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            # self.log("dis_loss_fake_train", d_loss_on_encoded)
            # return d_loss_on_encoded

        # if optimizer_idx == 1:
            ssim_img = self.ssim(watermark_img, img)
            d_on_encoded_for_enc = self.discriminator(watermark_img)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            watermark_text = self.extractor(watermark_img)
#             watermark_text = watermark_text + random_noise(watermark_text, -1, 1)
            output = sen[0,:]
            
            hidden = watermark_text[max_len, :, :]
            hidden = hidden.contiguous()
            intermediate_outputs = watermark_text[:max_len, :, :]
            
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, intermediate_outputs)
                outputs[t] = output
                teacher_force = random.random() < self.teacher_forcing
                top1 = output.max(1)[1]
                output = (sen[t] if teacher_force else top1)

            loss_img = self.mse_loss(img, watermark_img)

            loss_emb = self.mse_loss(encoder_outputs, watermark_text)

            text_out = outputs[1:].view(-1, outputs.shape[-1])
            # sen = sen[1:].view(-1)
            sen = sen[1:].reshape(-1)
            loss_text = self.loss_fn(text_out, sen)
            loss = 0.7*loss_img + 4*loss_emb + 10*loss_text + 1e-3*g_loss_adv
            # self.log("train_total_loss", loss)
            self.log_dict({"train_loss":loss, "train_img_loss":loss_img, "train_emb_loss":loss_emb, "train_text_loss":loss_text}, on_epoch=True)
            return loss
    
    # def training_epoch_end(self, training_step_outputs):
        # pdb.set_trace()
        # print("Training epoch output: ", training_step_outputs)
        # self.train_loss.append(training_step_outputs['train_loss'])

    def validation_step(self, batch, batch_idx):
        sen = batch[1].permute(1, 0)
        img = batch[0]
        batch_size = sen.shape[1]
        max_len = sen.shape[0]
        trg_vocab_size = self.decoder.output_dim
        cover_label = 1
        encoded_label = 0
        train_loss = []
        embedding_loss = []
        image_loss = []
        text_loss = []
        ssim_total = []
        adversarial_bce = []
        discr_cover_bce = []
        discr_encod_bce = []
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).float()
        outputs = outputs.type_as(sen)
        outputs = outputs.type(torch.float32)
        d_target_label_cover = torch.full((batch_size, 1), cover_label, dtype=torch.float32)
        d_target_label_cover = d_target_label_cover.type_as(sen)
        d_target_label_cover = d_target_label_cover.type(torch.float32)
        d_target_label_encoded = torch.full((batch_size, 1), encoded_label, dtype=torch.float32)
        d_target_label_encoded = d_target_label_encoded.type_as(sen)
        d_target_label_encoded = d_target_label_encoded.type(torch.float32)
        g_target_label_encoded = torch.full((batch_size, 1), cover_label, dtype=torch.float32)
        g_target_label_encoded = g_target_label_encoded.type_as(sen)
        g_target_label_encoded = g_target_label_encoded.type(torch.float32)

        # if optimizer_idx == 1:
        #     d_on_cover = self.discriminator(img)
        #     # pdb.set_trace()
        #     d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
        #     self.log("dis_loss_real_val", d_loss_on_cover, prog_bar=True)
        #     # print("Cover"*5)
        #     return d_loss_on_cover
        # else:
        encoder_outputs = self.encoder(sen)
        # print(img.shape, print(" main"*5))
        # if img.shape[0] != 16:
        #     pdb.set_trace()
        watermark_img = self.embedder(img, encoder_outputs)
        # train on fake
        d_on_encoded = self.discriminator(watermark_img)
        d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
        self.log("dis_loss_fake_val", d_loss_on_encoded)
        # return d_loss_on_encoded

    # if optimizer_idx == 1:
        ssim_img = self.ssim(watermark_img, img)
        d_on_encoded_for_enc = self.discriminator(watermark_img)
        g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
        watermark_text = self.extractor(watermark_img)
#         watermark_text = watermark_text + random_noise(watermark_text, -1, 1)
        output = sen[0,:]
        
        hidden = watermark_text[max_len, :, :]
        hidden = hidden.contiguous()
        intermediate_outputs = watermark_text[:max_len, :, :]
        
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, intermediate_outputs)
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing
            top1 = output.max(1)[1]
            output = (sen[t] if teacher_force else top1)

        loss_img = self.mse_loss(img, watermark_img)

        loss_emb = self.mse_loss(encoder_outputs, watermark_text)

        text_out = outputs[1:].view(-1, outputs.shape[-1])
        # sen = sen[1:].view(-1)
        sen = sen[1:].reshape(-1)
        loss_text = self.loss_fn(text_out, sen)
        loss = 0.7*loss_img + 4*loss_emb + 10*loss_text + 1e-3*g_loss_adv
        # self.log("val_total_loss", loss)
        self.log_dict({"val_loss":loss, "val_img_loss":loss_img, "val_emb_loss":loss_emb, "val_text_loss":loss_text}, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = [
                    {'params': self.encoder.parameters(), 'lr':1e-5},
                    {'params': self.decoder.parameters(), 'lr':1e-6},
                    {'params':self.embedder.parameters()},
                    {'params':self.extractor.parameters()}
                ]
        params_discrim = [{'params':self.discriminator.parameters()}]
        optimizer = optim.Adam(params, lr=1e-3, weight_decay= 1e-5)
        optimizer_discrim = optim.Adam(params_discrim, lr=1e-5)

        return [optimizer, optimizer_discrim]
    
    # def train_dataloader(self):
    #     '''https://github.com/Lightning-AI/lightning/issues/2457'''
    #     BATCH_SIZE=16
    #     en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    #     wordnet = WordNetLemmatizer()

    #     train_filepaths, val_filepaths, test_filepaths = download_data()
    #     en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    #     train_data = data_process(train_filepaths, en_vocab, en_tokenizer)
        
    #     train_dir = '/home/bishwa/v-5/imagenette/train/'
    #     train_remove = remove_image(train_dir)

    #     train_set = WatermarkDataset(train_dir, train_remove, train_data, en_vocab,
    #                                     transform_c=T.Compose([T.Resize(size=(256, 256)),T.ToPILImage()]), 
    #                                     data_size=7000
    #                                     )
    #     train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    #     return train_dataloader
        

def main():
    # en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    # wordnet = WordNetLemmatizer()

    # train_filepaths, val_filepaths, test_filepaths = download_data()
    # en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    # train_data = data_process(train_filepaths, en_vocab, en_tokenizer)
    # val_data = data_process(val_filepaths, en_vocab, en_tokenizer)
    # test_data = data_process(test_filepaths, en_vocab, en_tokenizer)

    BATCH_SIZE=8
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    wordnet = WordNetLemmatizer()

    train_filepaths, val_filepaths, test_filepaths = download_data()
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    train_data = data_process(train_filepaths, en_vocab, en_tokenizer)
    val_data = data_process(val_filepaths, en_vocab, en_tokenizer)
    train_dir = '/home/bishwa/v-5/imagenette/train/'
    val_dir = '/home/bishwa/v-5/imagenette/val/'
    train_remove = remove_image(train_dir)
    val_remove = remove_image(val_dir)
    
    train_set = WatermarkDataset(train_dir, train_remove, train_data, en_vocab,
                                    transform_c=T.Compose([T.Resize(size=(256, 256)),T.ToPILImage()]), 
                                    data_size=7000
                                    )
    val_set = WatermarkDataset(val_dir, val_remove, val_data, en_vocab,
                                    transform_c=T.Compose([T.Resize(size=(256, 256)),T.ToPILImage()]), 
                                    data_size=1000
                                    )
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    


    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(en_vocab)

    num_head = 2
    num_layers = 5
    hidden_size = 200
    emb_size = 270
    ENC_EMB_DIM = 64 #32
    DEC_EMB_DIM = 64 #32
    ENC_HID_DIM = 64 
    DEC_HID_DIM = 64 
    ATTN_DIM = 8
    ENC_DROPOUT = 0.6 #0.5
    DEC_DROPOUT = 0.6 #0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    discriminator = Discriminator()
    embedder = Embedder64()
    extractor = Extractor64()

    enc.load_state_dict(torch.load('/home/bishwa/v_6/saved_pretrained_model/encoder_13_noise_10p_bleu_94.pth'))
    dec.load_state_dict(torch.load('/home/bishwa/v_6/saved_pretrained_model/decoder_13_noise_10p_bleu_94.pth'))

    endtoend = EndtoEnd(enc, dec, embedder, extractor, discriminator)
    # trainer = pl.Trainer(max_epochs=10, gpus=3)
    csv_logger = CSVLogger(save_dir='./logs_end_to_end/',name='csv_file')
    trainer = pl.Trainer(devices=2, accelerator="gpu", max_epochs=100, strategy="ddp", logger=[csv_logger],)
    trainer.fit(endtoend, train_dataloader, val_dataloader)
    # trainer.fit(endtoend)

#https://stackoverflow.com/questions/70300576/how-to-access-loss-at-each-epoch-from-pytorch-lighting?rq=1
if __name__ == '__main__':
   main()