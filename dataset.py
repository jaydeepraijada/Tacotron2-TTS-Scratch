import librosa
import pandas as pd 
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tokenizer import Tokenizer
import numpy as np

def load_wav(path_to_audio, sr = 22050):
    audio, original_sr = torchaudio.load(path_to_audio)
    if original_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_freq=original_sr, new_freq=sr)
    return audio.squeeze(0)

def amp_to_db(x, min_db = -100):

    clip_val = 10 ** (min_db / 20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val))

def db_to_amp(x):
    return 10 ** (x / 20)


def normalize(x, min_db = -100, max_abs_val = 4):

    x = (x-min_db)/ -min_db
    x = x * 2 * max_abs_val - max_abs_val
    x = torch.clip(x, min = -max_abs_val, max = max_abs_val)

    return x

def denormalize(x, min_db = -100, max_abs_val = 4):

    x = torch.clip(x, min = -max_abs_val, max = max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val) 
    x = x * -min_db + min_db
     

    return x

class AudioMelConversion():

    def __init__(self, num_mels = 80, smapling_rate = 22050, n_fft = 1024, window_size = 1024, hop_size = 256, fmin=0, fmax=8000):

        self.num_mels = num_mels
        self.sampling_rate = smapling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = self.center
        self.min_db = self.min_db
        self.max_scaled_abs = self.max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)


    def _get_spec2mel_proj(self):
        mel = librosa.filters.mel(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)
        return torch.from_numpy(mel)

        
    def audio2mel(self, audio, do_norm = False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectogram = torch.stft(input = audio, n_fft = self.n_fft, hop_length = self.hop_size, win_length = self.window_size, window = torch.hann_window(self.window_size).to(audio.device), return_complex=True, pad_mode = 'reflect')
        spectogram = torch.abs(spectogram)
        mel = torch.matmul(self.spec2mel.to(audio.device), spectogram)

        mel = amp_to_db(mel, self.min_db)

        if do_norm:
            mel = normalize(mel, min_db = self.min_db, max_abs_val = self.max_scaled_abs)

        return mel
    
    def mel2audio(self, mel, do_denorm = False, griffin_lim_iters = 50):

        if do_denorm:
            mel = denormalize(mel, min_db = self.min_db, max_abs_val = self.max_scaled_abs)

        mel = db_to_amp(mel)

        spectogram = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()

        audio = librosa.griffinlim(spectogram, n_iter=griffin_lim_iters, hop_length=self.hop_size, win_length=self.window_size, window='hann')
        
        audio *= 32767/ max(0.01, np.max(np.abs(audio)))

        audio = audio.astype(np.int16)
        return audio
    
class TTSDataset(Dataset):

    def __init__(self, 
                 path_to_metadata, 
                 sample_rate = 22050, 
                 n_fft = 1024, 
                 window_size = 1024, 
                 hop_size = 256, 
                 num_mels = 80, 
                 fmin=0, 
                 fmax=8000,
                 center = False,
                 min_db = -100,
                 max_scaled_abs = 4):
        
        self.metadata = pd.read_csv(path_to_metadata)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.transcript_lengths = [len(Tokenizer().encode(transcript, return_tensor=False)) for transcript in self.metadata['transcript']]
        self.audio_proc = AudioMelConversion(num_mels = self.num_mels, smapling_rate = self.sample_rate, n_fft = self.n_fft, window_size = self.window_size, hop_size = self.hop_size, fmin=self.fmin, fmax=self.fmax)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        sample = self.metadata.iloc[index]

        path_to_audio = sample['file_path']
        transcript = sample['normalized_transcript']

        audio = load_wav(path_to_audio, sr = self.sample_rate)
        mel = self.audio_proc.audio2mel(audio, do_norm = True)

        return transcript, mel.squeeze(0)
    
def build_padding_mask(lengths):

    B = lengths.shape[0]
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1

    return mask.bool()
    
def TTSCollator():

    tokenizer = Tokenizer()

    def _collate_fn(batch):

        texts = [tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]

        input_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)

        
        input_lengths, sorted_idx = input_lengths.sort(descending=True)

        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

        mel_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]

        mel_padded = torch.zeros(len(mels), num_mels, mel_target_len)
        gate_padded = torch.zeros((len(mels), mel_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1

        mel_padded = mel_padded.transpose(1,2)

        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)   
    

class BatchSampler:

    def __init__(self, dataset, batch_size, drop_last = False):

        self.sample = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches()

    def _make_batches(self):

        indices = [i for i in self.sampler]

        if self.drop_last:

            total_size = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:total_size]

        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]  
        random_indices = torch.randperm(len(batches))

        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches:
            yield batch
    
    def __len__(self):
        return len(self.random_batches)
        

        
        
if __name__ == "__main__":
    

