# Self-transmit-and-receive spectrum analysis: generate high-frequency tx_n.wav from probe.wav,
# play tx_n.wav while recording to get rx_n.wav, then plot comparison spectrograms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import sounddevice as sd
from scipy.io import wavfile
import os
import threading
import time

# Font for labels (use DejaVu Sans if no CJK font)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. Parameters ---
# Your original voice command file
input_wav_file = 'probe.wav'
# Carrier frequency (Hz)
carrier_frequency = 20000.0
# Modulation depth (0.0 to 1.0)
modulation_depth = 1.0
# Output file sample rate (Hz)
output_sample_rate = 48000

# Recording flag
recording_finished = False
recorded_audio = None

def generate_modulated_signal():
    """Generate modulated signal and save as tx_n.wav"""
    print(f"Reading input file: {input_wav_file}")
    
    if not os.path.exists(input_wav_file):
        print(f"Error: Input file '{input_wav_file}' not found.")
        return None, None, None
    
    fs, audio_data = wavfile.read(input_wav_file)
    print(f"Input file sample rate: {fs} Hz")
    
    # If stereo, take one channel
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    
    # Normalize to [-1.0, 1.0]
    normalized_audio = audio_data / 32768.0
    
    # Generate modulated signal
    print(f"Generating modulated signal, carrier frequency {carrier_frequency} Hz...")
    
    num_samples_original = len(normalized_audio)
    duration = num_samples_original / fs
    num_samples_output = int(duration * output_sample_rate)
    t = np.linspace(0., duration, num_samples_output)
    
    # Resample audio to output sample rate
    resampled_audio = np.interp(
        t,
        np.linspace(0., duration, num_samples_original),
        normalized_audio
    )
    
    # Remove DC so audio is AC (important for AM)
    resampled_audio = resampled_audio - np.mean(resampled_audio)
    
    # Normalize so (1 + m*a(t)) stays non-negative
    audio_max = np.max(np.abs(resampled_audio))
    if audio_max > 0:
        resampled_audio = resampled_audio / audio_max  # normalize to [-1, 1]
    
    # Carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_frequency * t)
    
    # AM: s(t) = [1 + m*a(t)] * cos(2*pi*fc*t)
    # m = modulation depth, a(t) = normalized audio; yields carrier + upper/lower sidebands
    modulated_signal = (1 + modulation_depth * resampled_audio) * carrier_wave
    
    # Diagnostics
    print(f"Audio signal stats:")
    print(f"  Audio range: [{np.min(resampled_audio):.4f}, {np.max(resampled_audio):.4f}]")
    print(f"  Modulation envelope: [{np.min(1 + modulation_depth * resampled_audio):.4f}, {np.max(1 + modulation_depth * resampled_audio):.4f}]")
    
    # Normalize to avoid clipping
    modulated_signal /= np.max(np.abs(modulated_signal))
    
    output_data = np.int16(modulated_signal * 32767)
    
    tx_filename = f'tx_{int(carrier_frequency)}.wav'
    wavfile.write(tx_filename, output_sample_rate, output_data)
    print(f"Saved transmit signal: {tx_filename}")
    
    return modulated_signal, duration, resampled_audio

def record_audio(duration, sample_rate):
    """Record audio"""
    global recorded_audio, recording_finished
    
    record_duration = duration + 0.5
    print(f"Starting recording, duration: {record_duration:.2f} s...")
    recorded_audio = sd.rec(
        int(record_duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    recording_finished = True
    print("Recording done.")

def play_and_record(signal, duration, sample_rate):
    """Play and record simultaneously"""
    global recorded_audio, recording_finished
    
    recording_finished = False
    
    record_thread = threading.Thread(
        target=record_audio,
        args=(duration, sample_rate)
    )
    
    print("Preparing to play and record...")
    print("Starting in 3...")
    time.sleep(1)
    print("Starting in 2...")
    time.sleep(1)
    print("Starting in 1...")
    time.sleep(1)
    
    record_thread.start()
    
    print("Playing signal...")
    sd.play(signal, sample_rate)
    sd.wait()
    
    record_thread.join()
    
    return recorded_audio.flatten()

def plot_audio_spectrum(audio_file):
    """Plot frequency spectrum and spectrogram of audio"""
    print(f"\nAnalyzing audio file: {audio_file}")
    
    y, sr = librosa.load(audio_file, sr=None)
    
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} s")
    print(f"Sample count: {len(y)}")
    
    fig = plt.figure(figsize=(15, 10))
    
    # --- 1. Frequency spectrum ---
    ax1 = plt.subplot(2, 1, 1)
    
    N = len(y)
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, 1/sr)
    
    positive_freq_idx = xf >= 0
    xf_positive = xf[positive_freq_idx]
    yf_positive = yf[positive_freq_idx]
    
    magnitude = np.abs(yf_positive)
    
    ax1.plot(xf_positive, magnitude, linewidth=0.5)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.set_title('Frequency spectrum', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 40000)
    
    # --- 2. Spectrogram ---
    ax2 = plt.subplot(2, 1, 2)
    
    D = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
    magnitude_db_stft = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    img = librosa.display.specshow(magnitude_db_stft, 
                                    x_axis='time', 
                                    y_axis='hz', 
                                    sr=sr,
                                    hop_length=512,
                                    ax=ax2,
                                    cmap='viridis')
    
    ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    
    plt.colorbar(img, ax=ax2, format='%+2.0f dB', label='Magnitude (dB)')
    
    plt.tight_layout()
    
    output_file = f'rx_{int(carrier_frequency)}_spectrum.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Spectrum saved as: {output_file}")
    
    plt.show()

def plot_comparison_spectrum(probe_signal, tx_filename, recorded_signal, sample_rate):
    """Plot three spectra: probe.wav, modulated high-frequency carrier (from tx file), received signal"""
    print("\nPlotting comparison spectrum...")
    
    print(f"Reading modulated signal from file: {tx_filename}")
    if os.path.exists(tx_filename):
        fs_tx, tx_data = wavfile.read(tx_filename)
        if tx_data.ndim > 1:
            tx_data = tx_data[:, 0]
        modulated_signal = tx_data.astype(np.float32) / 32768.0
        print(f"Modulated signal from file: sample rate {fs_tx} Hz, length {len(modulated_signal)}")
    else:
        print(f"Warning: file {tx_filename} not found, using passed signal")
        modulated_signal = probe_signal
    
    # Align lengths (use shortest)
    min_len = min(len(probe_signal), len(modulated_signal), len(recorded_signal))
    probe_signal = probe_signal[:min_len]
    modulated_signal = modulated_signal[:min_len]
    recorded_signal = recorded_signal[:min_len]
    
    fig = plt.figure(figsize=(18, 6))
    
    # --- 1. probe.wav spectrum ---
    ax1 = plt.subplot(1, 3, 1)
    
    N_probe = len(probe_signal)
    probe_fft = np.fft.fft(probe_signal)
    xf_probe = np.fft.fftfreq(N_probe, 1/sample_rate)
    
    positive_freq_idx_probe = xf_probe >= 0
    xf_positive_probe = xf_probe[positive_freq_idx_probe]
    probe_magnitude = np.abs(probe_fft[positive_freq_idx_probe])
    
    ax1.plot(xf_positive_probe, probe_magnitude, linewidth=1.0, color='blue')
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Magnitude', fontsize=12)
    ax1.set_title('probe.wav spectrum', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 40000)
    
    # --- 2. Modulated high-frequency carrier spectrum ---
    ax2 = plt.subplot(1, 3, 2)
    
    N_modulated = len(modulated_signal)
    modulated_fft = np.fft.fft(modulated_signal)
    xf_modulated = np.fft.fftfreq(N_modulated, 1/sample_rate)
    
    positive_freq_idx_modulated = xf_modulated >= 0
    xf_positive_modulated = xf_modulated[positive_freq_idx_modulated]
    modulated_magnitude = np.abs(modulated_fft[positive_freq_idx_modulated])
    
    ax2.plot(xf_positive_modulated, modulated_magnitude, linewidth=1.0, color='green')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Modulated carrier spectrum', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 40000)
    
    # --- 3. Received signal spectrum ---
    ax3 = plt.subplot(1, 3, 3)
    
    N_recorded = len(recorded_signal)
    recorded_fft = np.fft.fft(recorded_signal)
    xf_recorded = np.fft.fftfreq(N_recorded, 1/sample_rate)
    
    positive_freq_idx_recorded = xf_recorded >= 0
    xf_positive_recorded = xf_recorded[positive_freq_idx_recorded]
    recorded_magnitude = np.abs(recorded_fft[positive_freq_idx_recorded])
    
    ax3.plot(xf_positive_recorded, recorded_magnitude, linewidth=1.0, color='red')
    ax3.set_xlabel('Frequency (Hz)', fontsize=12)
    ax3.set_ylabel('Magnitude', fontsize=12)
    ax3.set_title('Received signal (rx_n.wav) spectrum', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 40000)
    
    plt.tight_layout()
    
    output_file = f'comparison_spectrum_{int(carrier_frequency)}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison spectrum saved as: {output_file}")
    
    plt.show()

def main():
    """Main entry"""
    print("=== High-frequency play-and-record analysis ===")
    print(f"Carrier frequency: {carrier_frequency} Hz")
    print(f"Modulation depth: {modulation_depth}")
    print()
    
    signal, duration, probe_resampled = generate_modulated_signal()
    if signal is None:
        print("Could not generate modulated signal, exiting.")
        return
    
    recorded = play_and_record(signal, duration, output_sample_rate)
    
    rx_filename = f'rx_{int(carrier_frequency)}.wav'
    
    target_samples = int(duration * output_sample_rate)
    if len(recorded) > target_samples:
        recorded = recorded[:target_samples]
    
    if np.max(np.abs(recorded)) > 0:
        recorded_normalized = recorded / np.max(np.abs(recorded))
    else:
        recorded_normalized = recorded
    
    recorded_int16 = np.int16(recorded_normalized * 32767)
    
    wavfile.write(rx_filename, output_sample_rate, recorded_int16)
    print(f"Saved received signal: {rx_filename}")
    
    tx_filename = f'tx_{int(carrier_frequency)}.wav'
    plot_comparison_spectrum(probe_resampled, tx_filename, recorded_normalized, output_sample_rate)
    
    # Optional: plot received signal spectrum only
    # plot_audio_spectrum(rx_filename)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
