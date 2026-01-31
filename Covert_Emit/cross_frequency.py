import numpy as np
from scipy.io import wavfile
import os

# --- 1. Parameters ---
# Your original voice command file (ensure it exists in the script directory)
input_wav_file = 'probe.wav'
# Output "ultrasonic" attack signal file
output_wav_file = 'probe_20k.wav'

# Carrier frequency (Hz). We choose a frequency at the edge of human hearing but still playable by common speakers
# 17000 Hz is a good starting point, sounds like a sharp hiss
carrier_frequency = 20000.0

# Modulation depth (0.0 to 1.0). Use 1.0 for maximum effect
modulation_depth = 1.0

# Output file sample rate (Hz). 48000 Hz is common on macOS, sufficient for 17 kHz carrier
output_sample_rate = 48000

# Target output duration (seconds). Generated audio is looped to this duration
target_duration = 120.0  # two minutes

# --- Check if input file exists ---
if not os.path.exists(input_wav_file):
    print(f"Error: Input file '{input_wav_file}' not found.")
    print("Please prepare a short clear voice file and name it command.wav in this directory.")
else:
    # --- 2. Read and prepare baseband signal v(t) ---
    print(f"Reading input file: {input_wav_file}")
    fs, audio_data = wavfile.read(input_wav_file)
    print(f"Input file sample rate: {fs} Hz")

    # If stereo, take one channel
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Normalize to [-1.0, 1.0] (assuming 16-bit WAV)
    normalized_audio = audio_data / 32768.0

    # --- 3. Generate modulated signal s(t) ---
    print(f"Generating modulated signal, carrier frequency {carrier_frequency} Hz...")
    
    # Time axis for each sample
    num_samples_original = len(normalized_audio)
    duration = num_samples_original / fs
    num_samples_output = int(duration * output_sample_rate)
    t = np.linspace(0., duration, num_samples_output)

    # Resample original audio to new time axis via interpolation
    # Ensures voice and carrier sample points are aligned
    resampled_audio = np.interp(
        t,  # new time points
        np.linspace(0., duration, num_samples_original),  # original time points
        normalized_audio  # original signal
    )

    # Carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_frequency * t)

    # Amplitude modulation (AM)
    modulated_signal = (1 + modulation_depth * resampled_audio) * carrier_wave

    # --- 3.5. Loop signal to target duration ---
    print(f"Looping signal to {target_duration} s...")
    current_duration = len(modulated_signal) / output_sample_rate
    
    if current_duration < target_duration:
        # Loop repeat if signal is shorter than target
        num_repeats = int(np.ceil(target_duration / current_duration))
        modulated_signal = np.tile(modulated_signal, num_repeats)
        target_samples = int(target_duration * output_sample_rate)
        modulated_signal = modulated_signal[:target_samples]
        print(f"Signal looped {num_repeats} times, total duration: {len(modulated_signal) / output_sample_rate:.2f} s")
    elif current_duration > target_duration:
        target_samples = int(target_duration * output_sample_rate)
        modulated_signal = modulated_signal[:target_samples]
        print(f"Signal truncated to {target_duration} s")
    else:
        print(f"Signal duration already {target_duration} s, no change")

    # --- 4. Save output file ---
    print(f"Saving output file: {output_wav_file}")
    
    # Normalize modulated signal to avoid clipping
    modulated_signal /= np.max(np.abs(modulated_signal))

    # Convert to 16-bit integer for saving
    output_data = np.int16(modulated_signal * 32767)

    wavfile.write(output_wav_file, output_sample_rate, output_data)

    print("\nDone.")

