# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:51:00 2021

@author: johnny
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

# from os import path
from pydub import AudioSegment


src = r'Cranberries.mp3'
dst = "test.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")




y, sr = librosa.load(librosa.ex('test.wav'), duration=120)


# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

