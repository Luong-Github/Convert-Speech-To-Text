import collections
from concurrent.futures import thread
import ctypes
import io
import math
from zipfile import error
import numpy as num
import tempfile
import torch
import torch.nn.functional as F
from .utils import exact_div
import ffmpeg
import matplotlib as plt

from collections import namedtuple

import librosa

SAMPLERATE = 16000
N_FFT = 400 # fast furiour transform
N_MELS = 80 # mel-spectrogram 
HOP_LENGTH = 160 
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLERATE
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

def load_audio(file: str, sr: int = SAMPLERATE):
    """_summary_
    
    Open an audio file and read as mono waveform, resampling as neccessary

    Args:
        file (str): _description_: The audio file to opent
        sr (int, optional): _description_: the sample rate to resample the audio if necessary
    
    returns
        A numpy array containing the audio waveform, in float32type
    """
    
    "Should be change to use librosa"
    
    try:
        out, _ = (
            ffmpeg.input(file, threads = 0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stdeer=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError('')
    
    # devide to 32768.0 = 2^15 
    return num.frombuffer(out, num.int16).flatten().astype(num.float32) / 32768.0


def load_audio_librosa(file: str, sr: int = SAMPLERATE):
    """_summary_
    Using librosa to load file 

    Args:
        file (str): _description_
        sr (int, optional): _description_. Defaults to SAMPLERATE.

    Returns:
        tensor float 32: _description_
    """
    
    data, sr = librosa.load(file)
    
    # re sample rate to 16000
    data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLERATE)
    
    return data

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            
            
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = num.pad(array, pad_widths)

    return array

def normalize_audio(path):
    return (path)


