from scipy.io import wavfile
import scipy.signal as signal
import scipy.fft
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import numpy as np

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

class DspHelper:
    def __init__(self, win_length_ms, hop_length_ms, sampling_rate):
        self._win_length_ms = win_length_ms
        self._hop_length_ms = hop_length_ms
        self._sampling_rate = sampling_rate
        self._win_size_samples = int(np.around((self._win_length_ms/1000)*self._sampling_rate))
        self._hop_size_samples = int(np.around((self._hop_length_ms/1000)*self._sampling_rate))
        print(f"Window length {self._win_length_ms} ms = {self._win_size_samples} samples")
        print(f"Sampling rate {self._sampling_rate} Hz")    

    def windowing(self, data, windowing_function='hamming'):
        data = np.array(data)
        number_of_frames = 1 + int(np.floor((len(data)-self._win_size_samples) / self._hop_size_samples))
        frame_matrix = np.zeros((number_of_frames, self._win_size_samples))
    
        if windowing_function == 'rect':
            window = (np.sqrt(0.5))*np.ones((self._win_size_samples))
        elif windowing_function == 'hann':
            window = np.hanning(self._win_size_samples)
        elif windowing_function == 'cosine':
            window = np.sin(np.pi*((np.arange(self._win_size_samples)+0.5)/self._win_size_samples))
        elif windowing_function == 'hamming':
            window = np.hamming(self._win_size_samples)
        else:
            os.error('Windowing function not supported')    

        for i in range(number_of_frames):
            frame = np.zeros(self._win_size_samples)
            start = i*self._hop_size_samples
            stop = np.minimum(start+self._win_size_samples,len(data))
            frame[0:] = data[start:stop]
            frame_matrix[i,:] = np.multiply(window, frame)
        print(f"Constructed frame matrix {frame_matrix.shape[0]} * {frame_matrix.shape[1]}")
        return frame_matrix
    
    def _calculate_log_magnitude_spectrogram_for_frame(self, frame):
        eps = 1e-15
        fft_frame = np.fft.rfft(frame, len(frame))
        magnitude = np.abs(fft_frame)
        log_magnitude = 20.0*np.log(magnitude + eps)
        return log_magnitude

    def calculate_log_magnitude_spectrogram(self, frame_matrix):
        result = list()
        for frame in frame_matrix:
            fft_frame = self._calculate_log_magnitude_spectrogram_for_frame(frame)
            result.append(fft_frame)
        result = np.array(result)
        print(f"Constructed spectrogram {result.shape[0]} * {result.shape[1]}")
        return result

    def _calculate_log_mel_spectrogram_for_frame(self, frame, mel_filter_bank):
        eps = 1e-15
        fft_frame = np.fft.rfft(frame, len(frame))
        magnitude = np.abs(fft_frame)
        log_mel_spectrogram = 20*np.log10(np.matmul(np.transpose(mel_filter_bank), np.abs(fft_frame))+eps)        
        return log_mel_spectrogram #fft_frame #log_magnitude

    def calculate_log_mel_spectrogram(self, frame_matrix, mel_filter_bank):
        result = list()
        for frame in frame_matrix:
            fft_frame = self._calculate_log_mel_spectrogram_for_frame(frame, mel_filter_bank)
            result.append(fft_frame)
        result = np.array(result)
        return result

    def freq2mel(self,f): return 2595*np.log10(1 + (f/700))
    
    def mel2freq(self,m): return 700*(10**(m/2595) - 1)

    def build_mel_filter_bank(self, len_spectrum):
        melbands = 20 #20
        maxmel = self.freq2mel(8000)
        mel_idx = np.array(np.arange(.5,melbands,1)/melbands)*maxmel
        freq_idx = self.mel2freq(mel_idx)
        melfilterbank = np.zeros((len_spectrum,melbands))
        freqvec = np.arange(0,len_spectrum,1)*8000/len_spectrum
        for k in range(melbands-2):    
            if k>0:
                upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
            else:
                upslope = 1 + 0*freqvec
            if k<melbands-3:
                downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
            else:
                downslope = 1 + 0*freqvec
            triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
            melfilterbank[:,k] = triangle
        melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))
        return melfilterbank

    def calculate_mfcc(self, log_mel_spectogram):
        mfcc = scipy.fft.dct(log_mel_spectrogram)
        mfcc = mfcc[:,1 : 13] # probably correct
        #mfcc = mfcc[1 : 13,:] # wrong dimension
        
        #mfcc= 10 * np.log10(mfcc + 1e-15) #???
        return np.transpose(mfcc)


class PlotHelper:
    def plot(self, time_domain_signal, mel_filter_bank, spectrogram, log_mel_spectrogram, mfcc, spectrogram_torch, mel_spectrogram_torch, mfcc_torch):
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
        fig.tight_layout()
        plt.figure(1)

        plt.subplot(4,2,1)
        plt.plot(time_domain_signal)

        plt.subplot(4,2,2)
        plt.title("Mel filter bank")
        plt.plot(mel_filter_bank)
        
        plt.subplot(4,2,3)
        plt.imshow(spectrogram, origin='lower',aspect='auto',  extent=[0.,len(spectrogram), 0.,(16000/2)/1000])
        plt.title(f'Spectogram')
        plt.xlabel('Frame number')
        plt.ylabel('Frequency kHz')
       
        plt.subplot(4,2,4)
        plt.imshow(spectrogram_torch.numpy(), aspect='auto', origin='lower', extent=[0.,len(spectrogram_torch), 0.,(16000/2)/1000])
        plt.title("Spectogram using torch")
        plt.xlabel("Frame number")
        plt.ylabel("Frequency (kHz)")        

        plt.subplot(4,2,5)
        plt.title("Log mel spectrogram")
        plt.imshow(log_mel_spectrogram, origin='lower',aspect='auto',  extent=[0.,len(log_mel_spectrogram), 0.,(16000/2)/1000])

        plt.subplot(4,2,6)    
        plt.imshow(mel_spectrogram_torch.numpy(), aspect='auto', origin='lower',  extent=[0.,len(mel_spectrogram_torch), 0.,(16000/2)/1000])
        plt.title("Mel spectrogram using torch")
        plt.xlabel("Frame number)")
        plt.ylabel("Frequency (kHz)")
        
        plt.subplot(4,2,7)
        plt.title("MFCC")
        plt.imshow(mfcc, origin='lower',aspect='auto')

        plt.subplot(4,2,8)
        plt.title("MFCC using torch")
        plt.imshow(mfcc_torch, origin='lower',aspect='auto')

        plt.show()

# Read Data
data_dir = "../geo_ASR_challenge_2024"
file_helper = FileHelper(data_dir)
wav_file_list, text_label_list, sample_rate_list, data_list = file_helper.read_data("train.csv")

# Print Sample Data
index = 5
print(f"Sample {index}")
print(f"Wav file name {wav_file_list[index]}")
print(f"Text label {text_label_list[index]}")
print(f"Sample rate {sample_rate_list[index]}")
print(f"Speech signal length {len(data_list[index])}")

# Generate Dummy Submission file
test_wav_file_list, test_text_label_list, test_sample_rate_list, test_data_list = file_helper.read_data("test_release.csv")

for i in range(len(test_text_label_list)):
    test_text_label_list[i]  = "dummy speech recognition"
file_helper.write_submission_file("test_release.csv", test_wav_file_list, test_text_label_list)


# Calculate spectrogram, mel spectrogram, mfcc
time_domain_signal = data_list[index]
dsp_helper = DspHelper(win_length_ms=40, hop_length_ms=20, sampling_rate=sample_rate_list[index])
frame_matrix = dsp_helper.windowing(data_list[index])
spectrogram = dsp_helper.calculate_log_magnitude_spectrogram(frame_matrix)
mel_filter_bank = dsp_helper.build_mel_filter_bank(spectrogram.shape[1])
log_mel_spectrogram = dsp_helper.calculate_log_mel_spectrogram(frame_matrix, mel_filter_bank)
mfcc = dsp_helper.calculate_mfcc(log_mel_spectrogram)



import torch
import torchaudio
win_length = 640
hop_length = 320
n_fft = 640
sample_rate = 16000
spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2)

float_data_list = list()
for item in data_list[index]:
    float_data_list.append(float(item))
waveform = torch.from_numpy(np.array(float_data_list)) #.reshape(-1,1)
waveform = waveform.to(torch.float32)
spectrogram_torch = spectrogram_transform(waveform)
spectrogram_torch = 10 * torch.log10(spectrogram_torch + 1e-10)
spectrogram_torch = np.transpose(spectrogram_torch)

n_mels = 20
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)
mel_spectrogram_torch = mel_spectrogram_transform(waveform)
mel_spectrogram_torch = 10 * torch.log10(mel_spectrogram_torch + 1e-10)
mel_spectrogram_torch = np.transpose(mel_spectrogram_torch)


n_mfcc=13

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

mfcc_torch = mfcc_transform(waveform)

# Plot results
plot_helper = PlotHelper()
plot_helper.plot(time_domain_signal, mel_filter_bank, spectrogram, log_mel_spectrogram, mfcc, spectrogram_torch, mel_spectrogram_torch, mfcc_torch)





