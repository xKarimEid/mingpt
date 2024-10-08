"""
Script for downloading shakespear data
"""

import os
import requests
import torch
import tiktoken


class DataLoader:
    def __init__(self, train_conf):
        self.B = train_conf.B
        self.T = train_conf.T
        self.ddp_rank = train_conf.ddp_rank
        self.ddp_world_size  = train_conf.ddp_world_size
        self.current_start = self.B * self.T * self.ddp_rank

        self.data = self._read_encode_data()

        if train_conf.master_process:
            print(f"I am gpu: {train_conf.ddp_rank}")
            print(f"Number of tokens: {len(self.data)}")
            print(f"1 epoch: {len(self.data)//(self.B*self.T*self.ddp_world_size)}")

    def next_batch(self):
        buffer = self.data[self.current_start: 
                           1+ self.current_start + (self.B*self.T)]

        xb = buffer[:-1].view(self.B, self.T)
        yb = buffer[1:].view(self.B, self.T)
        
        # Updating current_start to new position
        self.current_start += self.B * self.T * self.ddp_world_size

        # Resetting if going over epoch
        if self.current_start + self.B*self.T* self.ddp_world_size + 1 > len(self.data):
            self.current_start = self.B * self.T * self.ddp_rank
        return xb, yb

    def _download_data(self):
        """Get file from internet and save it locally"""

        # Create the path where the data will be stored
        input_filepath = os.path.join(os.path.dirname('__file__'), 'input.txt')
        if not os.path.exists(input_filepath):
            #url for shakespear data
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            # Open and write to file
            with open(input_filepath, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)
    
    def _read_encode_data(self):
        # Download the data if not already done so
        self._download_data()
        # Read the data
        with open('input.txt') as f:
            data = f.read()
        
        # Load gp2 tokenizer
        enc = tiktoken.get_encoding('gpt2')
        data = enc.encode(data)
        data = torch.tensor(data)
        return data
