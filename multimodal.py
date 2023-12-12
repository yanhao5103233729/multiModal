# -*- coding: utf-8 -*-
# @time: Nov 13 2023
# @author: yanhao
# @software: PyCharm
"""
import os
import math
import gc
import joblib
import numpy as np
import pandas as pd
import scipy.io as scipyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as trn, models
from PIL import Image
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from datetime import datetime
from tqdm import tqdm

# Constants and Hyperparameters
RADAR_PARAMS = {
    'chirps': 128, 'tx': 1, 'rx': 4, 'samples': 256, 
    'adc_sampling': 5e6, 'chirp_slope': 15.015e12, 
    'start_freq': 77e9, 'idle_time': 5, 'ramp_end_time': 60
}

C = 3e8
chirp_period = (RADAR_PARAMS['ramp_end_time'] + RADAR_PARAMS['idle_time']) * 1e-6
RANGE_RES = C * RADAR_PARAMS['adc_sampling'] / (2 * RADAR_PARAMS['samples'] * RADAR_PARAMS['chirp_slope'])
VEL_RES_KMPH = 3.6 * C / (2 * RADAR_PARAMS['start_freq'] * chirp_period * RADAR_PARAMS['chirps'])

min_range_to_plot, max_range_to_plot = 5, 15
samples_per_chirp, n_chirps_per_frame = RADAR_PARAMS['samples'], RADAR_PARAMS['chirps']
acquired_range = samples_per_chirp * RANGE_RES
first_range_sample = np.ceil(samples_per_chirp * min_range_to_plot / acquired_range).astype(int)
last_range_sample = np.ceil(samples_per_chirp * max_range_to_plot / acquired_range).astype(int)
round_min_range, round_max_range = first_range_sample / samples_per_chirp * acquired_range, last_range_sample / samples_per_chirp * acquired_range

# Functions for range-velocity and range-angle maps
def range_velocity_map(data):
    data = np.fft.fft(data, axis=1)  # Range FFT
    data = np.fft.fft(data, axis=2)  # Velocity FFT
    data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis=0)  # Sum over antennas
    data = np.log(1 + data)
    return data

def range_angle_map(data, fft_size=64):
    data = np.fft.fft(data, axis=1)  # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis=0)  # Angle FFT
    data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis=2)  # Sum over velocity
    return data.T

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    # Implementation of Positional Encoding
    # ...

# Attention Class
class Attention(nn.Module):
    # Implementation of Attention mechanism
    # ...

# Cross Attention Class
class CrossAttention(nn.Module):
    # Implementation of Cross Attention mechanism
    # ...

# Custom Dataset Class
class MyDataset(Dataset):
    # Custom dataset for loading and processing data
    # ...

# Data Processing and Model Initialization
# ... (The rest of your existing code)

# Training and Validation Loop
# ... (Your training and validation code)

# Plotting Accuracy Curves
# ... (Your plotting code)

# Save the Trained Model
# ... (Your model saving code)
