import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab, vocab
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.metrics import bleu_score
import io

import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import pdb

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pytorch_lightning as pl
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
from functools import partial
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


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
  
    v = vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], special_first=True)
    v.set_default_index(0)
    return v

def data_process(filepaths, en_vocab, en_tokenizer):
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for raw_en in raw_en_iter:
        en_token = [en_vocab[token] for token in en_tokenizer(raw_en)][:13]
        if len(en_token) != 13:
            en_token.extend((13-len(en_token))*[1])
        en_tensor_ = torch.tensor(en_token,
                                dtype=torch.long)
        data.append((en_tensor_, en_tensor_))
    return data

def generate_batch(data_batch, BOS_IDX, EOS_IDX):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))

  de_batch = pad_sequence(de_batch, padding_value=1)
  en_batch = pad_sequence(en_batch, padding_value=1)
  return de_batch, en_batch


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
        self.layer_norm = nn.LayerNorm(self.emb_dim)
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
        hidden = self.layer_norm(hidden)
        outputs = self.layer_norm(outputs)
        hidden = torch.tanh(self.fc(hidden))
        outputs = torch.tanh(self.fc_2(outputs))
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
        self.layer_norm = nn.LayerNorm(self.emb_dim)
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
        output = self.layer_norm(output)
        decoder_hidden = self.layer_norm(decoder_hidden)
        weighted_encoder_rep = self.layer_norm(weighted_encoder_rep)

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)
    

class AutoEncoder(pl.LightningModule):
  def __init__(self, encoder, decoder, teacher_forcing=0.5):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=1)
    self.teacher_forcing = teacher_forcing
  
  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_index):
    src, trg = batch
    batch_size = src.shape[1]
    max_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim

    outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
    outputs = outputs.type_as(src)
    outputs = outputs.type(torch.float32)
    encoder_outputs = self.encoder(src)
    # encoder_outputs = encoder_outputs + random_noise(encoder_outputs, -1, 1)
    hidden = encoder_outputs[max_len, :, :]
    encoder_outputs = encoder_outputs[:max_len, :, :]
    output = trg[0,:]
    for t in range(1, max_len):
        output, hidden = self.decoder(output, hidden, encoder_outputs)
        outputs[t] = output
        teacher_force = random.random() < self.teacher_forcing
        top1 = output.max(1)[1]
        output = (trg[t] if teacher_force else top1)
    
    output = outputs[1:].view(-1, outputs.shape[-1])
    # output = output.type(torch.int64)
    trg = trg[1:].view(-1)
    # trg = trg.type(torch.int64)
    loss = self.loss_fn(output, trg)
    # loss = F.cross_entropy(output, trg)
    self.log("train_loss", loss, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_index):
    src, trg = batch
    batch_size = src.shape[1]
    max_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim

    outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
    outputs = outputs.type_as(src)
    outputs = outputs.type(torch.float32)
    encoder_outputs = self.encoder(src)
    # encoder_outputs = encoder_outputs + random_noise(encoder_outputs, -1, 1)
    hidden = encoder_outputs[max_len, :, :]
    encoder_outputs = encoder_outputs[:max_len, :, :]
    output = trg[0,:]
    for t in range(1, max_len):
        output, hidden = self.decoder(output, hidden, encoder_outputs)
        outputs[t] = output
        teacher_force = random.random() < self.teacher_forcing
        top1 = output.max(1)[1]
        output = (trg[t] if teacher_force else top1)
    
    output = outputs[1:].view(-1, outputs.shape[-1])
    trg = trg[1:].view(-1)
    loss = self.loss_fn(output, trg)
    # loss = F.cross_entropy(output, trg)
    self.log("val_loss", loss, on_epoch=True)
    return loss

  def test_step(self, batch, batch_idx):
    src, trg = batch
    batch_size = src.shape[1]
    max_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim

    outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
    outputs = outputs.type_as(src)
    outputs = outputs.type(torch.float32)
    encoder_outputs = self.encoder(src)
    encoder_outputs = encoder_outputs + random_noise(encoder_outputs, -2, 2)
    hidden = encoder_outputs[max_len, :, :]
    encoder_outputs = encoder_outputs[:max_len, :, :]
    output = trg[0,:]
    for t in range(1, max_len):
        output, hidden = self.decoder(output, hidden, encoder_outputs)
        outputs[t] = output
        teacher_force = random.random() < self.teacher_forcing
        top1 = output.max(1)[1]
        output = (trg[t] if teacher_force else top1)
    
    output = outputs[1:].view(-1, outputs.shape[-1])
    trg = trg[1:].view(-1)
    loss = self.loss_fn(output, trg)
    # loss = F.cross_entropy(output, trg)
    self.log("test_loss", loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer


def main():
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    wordnet = WordNetLemmatizer()

    train_filepaths, val_filepaths, test_filepaths = download_data()
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    train_data = data_process(train_filepaths, en_vocab, en_tokenizer)
    val_data = data_process(val_filepaths, en_vocab, en_tokenizer)
    test_data = data_process(test_filepaths, en_vocab, en_tokenizer)

    BATCH_SIZE = 256
    PAD_IDX = en_vocab['<pad>']
    BOS_IDX = en_vocab['<sos>']
    EOS_IDX = en_vocab['<eos>']

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=partial(generate_batch, BOS_IDX=BOS_IDX, EOS_IDX= EOS_IDX))
    valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=partial(generate_batch, BOS_IDX=BOS_IDX, EOS_IDX= EOS_IDX))
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=partial(generate_batch, BOS_IDX=BOS_IDX, EOS_IDX= EOS_IDX))
    

    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(en_vocab)

    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    ENC_HID_DIM = 64
    DEC_HID_DIM = 64
    ATTN_DIM = 8
    ENC_DROPOUT = 0.6
    DEC_DROPOUT = 0.6

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    csv_logger = CSVLogger(save_dir='./logs_pretraining/',name='csv_file')

    autoencoder = AutoEncoder(encoder=enc, decoder=dec)
    # trainer = pl.Trainer(accelerator="gpu", max_epochs=1, gpus=3, strategy="ddp")
    # trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer = pl.Trainer(devices=2, accelerator="gpu", max_epochs=100, strategy="ddp", logger=[csv_logger])
    trainer.fit(autoencoder, train_iter, valid_iter)


if __name__ == '__main__':
   main()