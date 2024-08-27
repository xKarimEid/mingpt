"""
Script for downloading shakespear data
"""


import os
import requests


# Create the path where the data will be stored
input_filepath = os.path.join(os.path.dirname('__file__'), 'input.txt')
if not os.path.exists(input_filepath):
    #url for shakespear data
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    # Open and write to file
    with open(input_filepath, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)
