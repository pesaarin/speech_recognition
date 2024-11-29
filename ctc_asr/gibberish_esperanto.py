from scipy.io import wavfile
#import scipy.signal as signal
#import scipy.fft
#from scipy.signal import lfilter
#import matplotlib.pyplot as plt
import numpy as np

#import torch
#import torchaudio

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
            data = np.float_(data)
            data_list.append(data.tolist())
        return sample_rate_list, data_list

    def read_data(self, csv_file):
        wav_file_list, text_label_list = self._read_csv_file(csv_file)
        sample_rate_list, data_list = self._read_wav_files(wav_file_list)
        
        #max_len = 0
        #for item in data_list:
        #    if len(item) > max_len:
        #        max_len = len(item)
        #print("max len", max_len)
        
        return wav_file_list, text_label_list, sample_rate_list, data_list

    def write_submission_file(self, csv_file, wav_file_list, text_label_list):
        number_of_samples = len(wav_file_list)
        f = open(csv_file, "w")
        f.write("file,transcript\n")
        for i in range(number_of_samples):
            f.write(f"{wav_file_list[i]},{text_label_list[i]}\n")
        f.close()


