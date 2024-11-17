from scipy.io import wavfile
import scipy.signal as signal
import scipy.fft
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchaudio

class FileHelper:
    def __init__(self, data_dir):
        self._data_dir = data_dir
        
    def _read_csv_file(self, csv_file):
        file = open(self._data_dir+"/"+csv_file, "r")
        content=file.readlines()
        file.close()
        wav_file_list = list()
        text_label_list = list()
        line_index = 0
        print(f"Reading data from {csv_file}. First samples and labels:")
        for line in content:
            if line_index == 0:
                line_index += 1
                continue; # first line not interesting
            comma_index = line.find(",")
            wav_file = line[:comma_index]
            text_label = line[comma_index+1:len(line)-1]
            wav_file_list.append(wav_file)
            text_label_list.append(text_label)
            line_index += 1
        print(f"Read {line_index-1} samples")
        return wav_file_list, text_label_list

    def _read_wav_files(self, wav_file_list):
        sample_rate_list = list()
        data_list = list()
        for wav_file in wav_file_list:
            sample_rate, data = wavfile.read(self._data_dir+"/"+wav_file)
            if sample_rate != 16000:
                print(f"{wav_file} has unexpected sample rate {sample_rate}")
            sample_rate_list.append(sample_rate)
            data_list.append(data)
        return sample_rate_list, data_list

    def read_data(self, csv_file):
        wav_file_list, text_label_list = self._read_csv_file(csv_file)
        sample_rate_list, data_list = self._read_wav_files(wav_file_list)
        return wav_file_list, text_label_list, sample_rate_list, data_list

    def write_submission_file(self, csv_file, wav_file_list, text_label_list):
        number_of_samples = len(wav_file_list)
        f = open(csv_file, "w")
        f.write("file,transcript\n")
        for i in range(number_of_samples):
            f.write(f"{wav_file_list[i]},{text_label_list[i]}\n")
        f.close()

    def convert_waveform_to_mfcc(self, waveform):
        waveform_as_floats = list()
        for item in waveform:
            waveform_as_floats.append(float(item))
        waveform = torch.from_numpy(np.array(waveform_as_floats))
        waveform = waveform.to(torch.float32)
    
        win_length = 640
        hop_length = 320
        n_fft = 640
        sample_rate = 16000
        n_mels = 40
        n_mfcc = 12

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'center': True
            }
        )

        mfcc = mfcc_transform(waveform)
        mfcc = torch.transpose(mfcc, 0, 1)
        return mfcc

    def convert_data(self, data_list):
        mfcc_list = list()
        for item in data_list:
            mfcc = self.convert_waveform_to_mfcc(item)
            mfcc_list.append(mfcc)
        return mfcc_list

    def convert_labels(self, text_label_list):
        converted_list = list()
        for item in text_label_list:
            sub_list = list()
            sub_list.append(item)
            converted_list.append(sub_list)
        return converted_list

    def read(self, csv_file):
        wav_file_list, text_label_list, sample_rate_list, data_list = self.read_data(csv_file)
        mfcc_list = list()
        for item in data_list:
            mfcc = self.convert_waveform_to_mfcc(item)
            mfcc_list.append(mfcc)
        mfcc_list = self.convert_data(data_list)
        label_list = self.convert_labels(text_label_list)
        return mfcc_list, label_list, wav_file_list

