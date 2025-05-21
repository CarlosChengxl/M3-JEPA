# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pandas as pd
import csv
import json
import subprocess
import time

import numpy as np
from PIL import Image
from logging import getLogger
import soundfile as sf
from scipy.io import wavfile


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset as HFDataset, load_dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor
import src.models.model_paths as model_paths
_GLOBAL_SEED = 0
logger = getLogger()

#不用
class DatasetManager_speech:
    def __init__(self, batch_size, pin_mem=True, num_workers=8, world_size=1, rank=0, tsv_path=None, audio_folder=None, val_split=0.8, speech_encoder_name=None, text_encoder_name=None):
        self.batch_size = batch_size
        self.pin_mem = pin_mem
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank
        self.tsv_path = tsv_path
        self.audio_folder = audio_folder
        self.val_split = val_split
        self.speech_encoder_name = speech_encoder_name
        self.text_encoder_name = text_encoder_name

        self.dataset = self.make_dataset()

    def make_dataset(self):
        # print("datamanager:", self.tsv_path)
        return audiocapsDataset(tsv_path=self.tsv_path, audio_folder=self.audio_folder, mode='train', speech_encoder_name = self.speech_encoder_name, text_encoder_name = self.text_encoder_name)

    def resplit_data(self):
        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_dist_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        val_dist_sampler = DistributedSampler(
            dataset=val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )

        train_data_loader = DataLoader(
            dataset=train_dataset,
            sampler=train_dist_sampler,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=self.pin_mem,
            num_workers=self.num_workers,
            persistent_workers=False
        )

        val_data_loader = DataLoader(
            dataset=val_dataset,
            sampler=val_dist_sampler,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=self.pin_mem,
            num_workers=self.num_workers,
            persistent_workers=False
        )

        return train_dataset, val_dataset, train_data_loader, val_data_loader, train_dist_sampler, val_dist_sampler

##train dataset
def make_speech_train_set(
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    tsv_path=None,
    audio_folder=None,
    speech_encoder_name=None,
    text_encoder_name=None
):
    train_set = make_train_dataset(   
        root_path=root_path,
        tsv_path=tsv_path,
        audio_folder=audio_folder,
        speech_encoder_name=speech_encoder_name,
        text_encoder_name=text_encoder_name)

    train_dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        sampler=train_dist_sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    return train_set, train_data_loader, train_dist_sampler

def make_train_dataset(
    root_path=None,
    tsv_path=None,
    audio_folder=None,
    speech_encoder_name=None,
    text_encoder_name=None
):
    train_dataset = audiocapsDataset(
        tsv_path=tsv_path,
        audio_folder=audio_folder,
        mode='train',
        speech_encoder_name = speech_encoder_name,
        text_encoder_name = text_encoder_name)
    return train_dataset




##########val dataset
def make_speech_val_set(
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    tsv_path=None,
    audio_folder=None,
    speech_encoder_name=None,
    text_encoder_name=None
):
    val_set = make_val_dataset(   
        root_path=root_path,
        tsv_path=tsv_path,
        audio_folder=audio_folder,
        speech_encoder_name=speech_encoder_name,
        text_encoder_name=text_encoder_name)

    val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False)

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        sampler=val_dist_sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    return val_set, val_data_loader, val_dist_sampler

def make_val_dataset(
    root_path=None,
    tsv_path=None,
    audio_folder=None,
    speech_encoder_name=None,
    text_encoder_name=None
):
    val_dataset = audiocapsDataset(
        tsv_path=tsv_path,
        audio_folder=audio_folder,
        mode='val',
        speech_encoder_name = speech_encoder_name,
        text_encoder_name = text_encoder_name)
    return val_dataset





####################################



def make_speech_test_set(
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    tsv_path=None,
    audio_folder=None,
    speech_encoder_name=None,
    text_encoder_name=None
):
    test_set = make_test_dataset(   
        root_path=root_path,
        tsv_path=tsv_path,
        audio_folder=audio_folder,
        speech_encoder_name=speech_encoder_name,
        text_encoder_name=text_encoder_name)

    test_dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=test_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        sampler=test_dist_sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    return test_set, test_data_loader, test_dist_sampler

def make_test_dataset(
    root_path=None,
    tsv_path=None,
    audio_folder=None,
    speech_encoder_name=None,
    text_encoder_name=None
):
    test_dataset = audiocapsDataset(
        tsv_path=tsv_path,
        audio_folder=audio_folder,
        mode='test',
        speech_encoder_name = speech_encoder_name,
        text_encoder_name = text_encoder_name)
    return test_dataset

def downsample_audio_mask(audio_mask):
    audio_mask_np = audio_mask.cpu().numpy()[0]
    downsampled_mask = []
    for i in range(0, len(audio_mask_np), 2):
        if audio_mask_np[i] == 1 or audio_mask_np[i+1] == 1:
            downsampled_mask.append(1)
        else:
            downsampled_mask.append(0)
    
    downsampled_mask = torch.tensor([downsampled_mask], device=audio_mask.device, dtype=torch.int32)
    return downsampled_mask

class audiocapsDataset(Dataset):
    def __init__(self,
                 tsv_path,
                 audio_folder,
                 mode='train',
                 speech_encoder_name = None,
                 text_encoder_name = None):

        
        self.tsv_path = tsv_path
        self.audio_folder = audio_folder
        self.mode = mode
        logger.info(f"Load Language model: {text_encoder_name}. Load audio model: {speech_encoder_name}")

        model_id = "/vepfs/DI/beijing-public/models/whisper/whisper-large-v3"
        self.audio_processor = AutoProcessor.from_pretrained(model_id)
        # self.audio_processor = AutoProcessor.from_pretrained(model_paths.sm_model_path[speech_encoder_name])
        self.text_processor = AutoTokenizer.from_pretrained(model_paths.lm_model_path[text_encoder_name])
        self.text_processor.pad_token = self.text_processor.eos_token


        if mode == 'train':
            suffix = 'audiocaps_train_wav.tsv'
            settype = 'train_wav/'
            self.audio_folder = self.audio_folder + settype
        elif mode == 'val':
            suffix = 'audiocaps_val_wav.tsv'
            settype = 'val_wav/'
            self.audio_folder = self.audio_folder + settype
        else:
            suffix = 'audiocaps_test_wav.tsv'
            settype = 'test_wav/'
            self.audio_folder = self.audio_folder + settype

        # 读取tSV文件
        # print("tsv路径：",tsv_path)
        path = tsv_path + suffix
        df = pd.read_csv(path, sep='\t')

        logger.info(f'Initialized audiocapsDataset with {len(df)} audio for {mode} mode.')
        
        # 将Pandas DataFrame转换为HuggingFace Dataset格式
        self.dataset = HFDataset.from_pandas(df)
        

    def __len__(self):
        return len(self.dataset)
            
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_path = self.audio_folder + item['audio']
        # print("audio_path:",type(audio_path))
        # data = load_dataset(audio_path)
        # print(audio_path)
        # data, samplerate = sf.read(audio_path)

        # try:
        #     samplerate, data = wavfile.read(audio_path)
        #     print(f"Successfully read {audio_path}")
        # except Exception as e:
        #     print(f"Failed to read {audio_path}. Error: {e}")

        samplerate, data = wavfile.read(audio_path)

        #inputs = self.audio_processor(data, sampling_rate=samplerate, nb_max_frams=1000, return_tensors="pt")
        inputs = self.audio_processor(data, sampling_rate=samplerate, nb_max_frams=1000, return_tensors="pt", return_attention_mask=True)
        # inputs = self.audio_processor(data["audio"]["array"], sampling_rate=16000, return_tensors="pt")
        audio = inputs.input_features.squeeze(dim=0)
        audio_mask = inputs.attention_mask
        audio_mask_downsampled = downsample_audio_mask(audio_mask)
        audio_mask_downsampled = audio_mask_downsampled.squeeze(dim=0)
        
        # print("audio_shape:",audio.shape)
        # print("audio_mask_shape:",audio_mask.shape)
        # print("number of 1s:", audio_mask.eq(1).sum().item())
        # print("number of 0s:", audio_mask.eq(0).sum().item())
        # print("audio_mask_downsampled_shape:",audio_mask_downsampled.shape)
        # print("number of 1s:", audio_mask_downsampled.eq(1).sum().item())
        # print("number of 0s:", audio_mask_downsampled.eq(0).sum().item())
        captions = item['text']  # 获取所有描述
        sentids = item['uniq_id']    # 获取所有句子ID
        processed_captions = self.text_processor(
            captions, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        input_ids = processed_captions['input_ids'].squeeze(dim=0)
        attention_mask = processed_captions['attention_mask'].squeeze(dim=0)
        
        # return {
        #     'audio_features': audio,
        #     'text_features': {
        #         'input_ids': input_ids,
        #         'attention_mask': attention_mask
        #     }
        return {
            'audio_features': {
                'audio': audio,
                'audio_mask': audio_mask_downsampled
            },
            'text_features': {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        } 




