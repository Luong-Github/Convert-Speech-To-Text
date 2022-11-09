import numpy as num
from multiprocessing import Pool
import torch
import torch.nn.functional as F
from .utils import exact_div
import ffmpeg
import matplotlib as plt

from scipy.ndimage.morphology import binary_dilation

from tqdm import tqdm
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union
from functools import partial

import struct

from config import librispeech_config
from warnings import warn

import librosa

## Audio volume normalization
audio_norm_target_dBFS = -30
SAMPLERATE = 16000

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40

vad_window_length = 30  # In milliseconds
vad_moving_average_width = 8

sampling_rate = 16000
N_FFT = 400 # fast furiour transform
N_MELS = 80 # mel-spectrogram 
HOP_LENGTH = 160 
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLERATE
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)
_AUDIO_EXTENSIONS = ("wav", "flac", "m4a", "mp3")

int16_max = (2 ** 15) - 1

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

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



def preprocess_wav(fpath_or_wav: Union[str, Path, num.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.
    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        wav,
        sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(num.float32).T


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(num.round(wav * int16_max)).astype(num.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = num.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = num.concatenate((num.zeros((width - 1) // 2), array, num.zeros(width // 2)))
        ret = num.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = num.round(audio_mask).astype(num.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, num.ones(6 + 1))
    audio_mask = num.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * num.log10(num.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


"""
Region

"""

def _preprocess_speaker(speaker_dir: Path, datasets_root: Path, out_dir: Path, skip_existing: bool):
    # Give a name to the speaker that includes its dataset
    speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)

    # Create an output directory with that name, as well as a txt file containing a
    # reference to each source file.
    speaker_out_dir = out_dir.joinpath(speaker_name)
    speaker_out_dir.mkdir(exist_ok=True)
    sources_fpath = speaker_out_dir.joinpath("_sources.txt")

    # There's a possibility that the preprocessing was interrupted earlier, check if
    # there already is a sources file.
    if sources_fpath.exists():
        try:
            with sources_fpath.open("r") as sources_file:
                existing_fnames = {line.split(",")[0] for line in sources_file}
        except:
            existing_fnames = {}
    else:
        existing_fnames = {}

    # Gather all audio files for that speaker recursively
    sources_file = sources_fpath.open("a" if skip_existing else "w")
    audio_durs = []
    for extension in _AUDIO_EXTENSIONS:
        for in_fpath in speaker_dir.glob("**/*.%s" % extension):
            # Check if the target output file already exists
            out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, ".npy")
            if skip_existing and out_fname in existing_fnames:
                continue

            # Load and preprocess the waveform
            wav = preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue

            # Create the mel spectrogram, discard those that are too short
            frames = wav_to_mel_spectrogram(wav)
            # partials_n_frames = 160
            if len(frames) < 160:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            num.save(out_fpath, frames)
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))
            audio_durs.append(len(wav) / sampling_rate)

    sources_file.close()

    return audio_durs

def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger):
    print("%s: Preprocessing data for %d speakers." % (dataset_name, len(speaker_dirs)))

    # Process the utterances for each speaker
    work_fn = partial(_preprocess_speaker, datasets_root=datasets_root, out_dir=out_dir, skip_existing=skip_existing)
    with Pool(4) as pool:
        tasks = pool.imap(work_fn, speaker_dirs)
        for sample_durs in tqdm(tasks, dataset_name, len(speaker_dirs), unit="speakers"):
            for sample_dur in sample_durs:
                logger.add_sample(duration=sample_dur)

    logger.finalize()
    print("Done preprocessing %s.\n" % dataset_name)

def init_processing_dataset(dataset_name, dataset_root, out_dir) -> (Path):
    dataset_root = dataset_root.joinpath(dataset_name)
    if not dataset_root.exists():
        print("Couldn\'t find %s, skipping this dataset." % dataset_root)
        return None, None
    return dataset_root, ""

def prepocessing_librispeech(dataset_root: Path, out_dir: Path, skip_existing=False):
    
    for dataset_name in librispeech_config["train"]["clean"]:
        dataset_root, logger = init_processing_dataset(dataset_name, dataset_root, out_dir)
        
        if not dataset_root:
            return

        speaker_dirs = list(dataset_root.glob('*'))
        _preprocess_speaker_dirs(speaker_dirs, dataset_name, dataset_root, out_dir, skip_existing, logger)
        
    
