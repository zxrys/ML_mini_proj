import os
import librosa
import numpy as np
import scipy.stats
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Tokenizer
import joblib

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义音频加载和预处理函数
def load_audio(file_path, duration=300, sr=16000):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        audio = librosa.util.normalize(audio)
        desired_length = sr * duration
        if len(audio) > desired_length:
            audio = audio[:desired_length]
        elif len(audio) < desired_length:
            padding = desired_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# 定义特征提取函数
def extract_mfcc(audio, sr=16000, n_mfcc=13, n_fft=400, hop_length=160):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc


def extract_prosodic_features(audio, sr=16000):
    pitches, voiced_flags, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    pitch_mean = np.nanmean(pitches)
    pitch_std = np.nanstd(pitches)
    rms = librosa.feature.rms(y=audio)
    energy_mean = np.mean(rms)
    energy_std = np.std(rms)
    zero_crossings = librosa.zero_crossings(audio, pad=False)
    speech_rate = np.sum(zero_crossings) / len(audio) * sr
    return {
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'speech_rate': speech_rate
    }


def extract_spectral_features(audio, sr=16000):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    return {
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_std': np.std(spectral_bandwidth),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'spectral_rolloff_std': np.std(spectral_rolloff)
    }


# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的Wav2Vec 2.0模型和tokenizer，并将模型移动到GPU（如果可用）
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)


def extract_wav2vec_embeddings(audio, sr=16000):
    input_values = tokenizer(audio, return_tensors="pt", sampling_rate=sr).input_values.to(device)
    with torch.no_grad():
        embeddings = wav2vec_model(input_values).last_hidden_state
    embeddings = embeddings.mean(dim=1).squeeze().cpu().numpy()
    return embeddings


def extract_vggish_embeddings(audio, sr=16000):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    if log_mel_spectrogram.shape[1] < 96:
        pad_width = 96 - log_mel_spectrogram.shape[1]
        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :96]
    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)
    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)
    # Placeholder embeddings
    embeddings = np.random.rand(128)
    return embeddings


def aggregate_features(mfcc, prosodic, spectral, wav2vec, vggish):
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_skew = scipy.stats.skew(mfcc, axis=1)
    mfcc_kurt = scipy.stats.kurtosis(mfcc, axis=1)

    prosodic_features = np.array(list(prosodic.values()))
    spectral_features = np.array(list(spectral.values()))

    wav2vec_features = np.array(wav2vec)
    vggish_features = np.array(vggish)

    # 验证所有特征是一维的
    assert mfcc_mean.ndim == 1, f"mfcc_mean has {mfcc_mean.ndim} dimensions"
    assert mfcc_std.ndim == 1, f"mfcc_std has {mfcc_std.ndim} dimensions"
    assert mfcc_skew.ndim == 1, f"mfcc_skew has {mfcc_skew.ndim} dimensions"
    assert mfcc_kurt.ndim == 1, f"mfcc_kurt has {mfcc_kurt.ndim} dimensions"
    assert prosodic_features.ndim == 1, f"prosodic_features has {prosodic_features.ndim} dimensions"
    assert spectral_features.ndim == 1, f"spectral_features has {spectral_features.ndim} dimensions"
    assert wav2vec_features.ndim == 1, f"wav2vec_features has {wav2vec_features.ndim} dimensions"
    assert vggish_features.ndim == 1, f"vggish_features has {vggish_features.ndim} dimensions"

    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurt,
        prosodic_features,
        spectral_features,
        wav2vec_features,
        vggish_features
    ])
    return feature_vector


def extract_features(audio_files):
    """Extract and aggregate features from a list of audio files."""
    feature_matrix = []
    for idx, audio in enumerate(audio_files):
        print(f"Processing audio file {idx + 1}/{len(audio_files)}")
        mfcc = extract_mfcc(audio)
        prosodic = extract_prosodic_features(audio)
        spectral = extract_spectral_features(audio)
        wav2vec = extract_wav2vec_embeddings(audio)
        vggish = extract_vggish_embeddings(audio)
        aggregated = aggregate_features(mfcc, prosodic, spectral, wav2vec, vggish)
        feature_matrix.append(aggregated)
    return np.array(feature_matrix)


def get_audio_file_paths(audio_directory, extension='.wav'):
    """Retrieve all audio file paths with the given extension."""
    return [
        os.path.join(audio_directory, file)
        for file in os.listdir(audio_directory)
        if file.endswith(extension)
    ]


def load_audio_files(audio_file_paths):
    """Load audio files from the given file paths."""
    audio_files = []
    for file_path in audio_file_paths:
        audio = load_audio(file_path)
        if audio is not None:
            audio_files.append(audio)
    return audio_files


def generate_labels(num_samples, seed=42):
    """Generate random labels as a placeholder. Replace with actual labels."""
    np.random.seed(seed)
    return np.random.randint(0, 2, size=num_samples)


def save_features(feature_matrix, labels, feature_path, label_path):
    """Save feature matrix and labels to disk."""
    joblib.dump(feature_matrix, feature_path)
    joblib.dump(labels, label_path)
    print(f"Features saved to {feature_path}")
    print(f"Labels saved to {label_path}")


def load_features(feature_path, label_path):
    """Load feature matrix and labels from disk."""
    feature_matrix = joblib.load(feature_path)
    labels = joblib.load(label_path)
    print(f"Features loaded from {feature_path}")
    print(f"Labels loaded from {label_path}")
    return feature_matrix, labels


def process_and_save_features(audio_directory, feature_path, label_path):
    """Full pipeline to process audio files and save features and labels."""
    audio_file_paths = get_audio_file_paths(audio_directory)
    print(f"Found {len(audio_file_paths)} audio files.")

    audio_files = load_audio_files(audio_file_paths)
    print(f"Successfully loaded {len(audio_files)} audio files.")

    feature_matrix = extract_features(audio_files)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    labels = generate_labels(len(audio_files))

    save_features(feature_matrix, labels, feature_path, label_path)


if __name__ == "__main__":
    audio_directory = '../../data/raw/CBU0521DD_stories'
    feature_path = '../../data/processed/audio_feature/feature_matrix.joblib'
    label_path = '../../data/processed/audio_feature/labels.joblib'

    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

    process_and_save_features(audio_directory, feature_path, label_path)
