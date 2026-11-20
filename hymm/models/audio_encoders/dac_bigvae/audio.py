import io
import random

import torch
import torchaudio
# import whisper
from librosa.filters import mel as librosa_mel_fn
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F

def load_and_resample_if_necessary(audio_bytes, target_sr):
    waveform, sr = load(audio_bytes)
    if sr != target_sr:
        waveform = resample(waveform, sr, target_sr)

    wav_len = waveform.size(0)
    duration = wav_len / target_sr

    return waveform, wav_len, duration


def load(audio_bytes):
    if type(audio_bytes) == str:
        waveform, sample_rate = torchaudio.load(audio_bytes)
        waveform = waveform[0, :]
    else:
        with io.BytesIO(audio_bytes) as f:
            waveform, sample_rate = torchaudio.load(f)
            waveform = waveform[0, :]

    return waveform, sample_rate

def save(uri, src, sample_rate, channels_first=True):
    torchaudio.save(uri, src.unsqueeze(0), sample_rate, channels_first)


resample_transforms = {}


def resample(waveform, orig_sr, target_sr=24000):
    if orig_sr == target_sr:
        return waveform

    global resample_transforms
    k = f"{orig_sr}_{target_sr}"
    if k not in resample_transforms:
        resample_transforms[k] = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr
        )

    waveform = resample_transforms[k](waveform)
    waveform = torch.clamp(waveform, min=-1.0, max=1.0)

    return waveform


def compute_prompt_mel(waveform, max_prompt_audio_seconds):
    mel = compute_mel(waveform.unsqueeze(0))
    mel = mel.squeeze(0)

    prompt_mel, _, _ = truncate_mel(mel, max_prompt_audio_seconds)
    return prompt_mel


def compute_mel(
    y,
    n_fft=1024,
    num_mels=100,
    sampling_rate=24000,
    hop_size=256,
    win_size=1024,
    fmin=0,
    fmax=12000,
    center=False,
):
    """
    Reference: https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py#L60
    """
    assert torch.min(y) >= -1.0 and torch.max(y) <= 1.0

    mel_basis, hann_window = {}, {}
    str_key_mel_basis = f"{str(fmax)}_{str(y.device)}"
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str_key_mel_basis] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str_key_mel_basis], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if fmax is None:
        fmax = sampling_rate / 2
    mel_basis, hann_window = {}, {}
    if fmax not in mel_basis:
        mel = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, fmin, fmax, num_mels, sampling_rate).T
        mel_basis[str(fmax) + '_' + str(y.device)] = mel.to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y, (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    mag_spec = torch.abs(spec)
    mel_spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], mag_spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def bigvgan_wav_to_mel(
    y,
    n_fft=1024,
    num_mels=100,
    sampling_rate=24000,
    hop_size=256,
    win_size=1024,
    fmin=0,
    fmax=12000,
    center=False,
) -> torch.Tensor:
    """
    Args: y shape [B, T]

    Return: [B, num_mels, D]
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    mel_basis, hann_window = {}, {}
    if fmax not in mel_basis:

        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )

        # mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def truncate_mel(mel, max_audio_seconds):
    assert mel.ndim == 2 and mel.size(0) == 100

    wanted_mel_length = int(max_audio_seconds * 24000 // 256)

    if mel.size(1) > wanted_mel_length:
        max_offset = mel.size(1) - wanted_mel_length
        start = random.randint(0, max_offset - 1)
        end = start + wanted_mel_length
        return mel[:, start:end], int(start // 4), int(end // 4)

    start = 0
    end = mel.size(1)
    return mel, int(start // 4), int(end // 4)


# def compute_whisper_mel(waveform):
#     mel_len = waveform.size(0) // whisper.audio.HOP_LENGTH
#     audio = whisper.pad_or_trim(waveform)
#     mel = whisper.log_mel_spectrogram(audio)
#     return mel, mel_len

def compute_wenet_mel(waveform):
    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10
    dither = 0.0
    sample_rate = 16000
    mel = kaldi.fbank(waveform.unsqueeze(0) * (1 << 15),
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        energy_floor=0.0,
                        sample_frequency=sample_rate)
    return mel.T, mel.size(0)


class CampPlusFBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr == self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat