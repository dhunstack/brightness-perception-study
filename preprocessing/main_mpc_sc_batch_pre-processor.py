import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


'''
------------------------------ mcp.Spectral Centroid Batch-Pre-Processor ------------------------------

- Asks for File or Folder Path
- For each File in Folder:
    - Calculates Spectral Centroid
    - Adds magnitude offset above and reduction below Spectral Centroid
    - Recalculates Spectral Centroid and Delta after Processing
    - Write adjusted files to folder
    
Author: Robin Doerfler
Project: MPC - Course Team Project - JND for Brightness / Spectral Centroid
Date: 2023-11-20

'''


def get_base_path():
    # Get Folder contents
    input_path = input("1. Copy File / Folder Path"
                       "\n"
                       "[option + cmnd + c to copy path of highlightet item on MAC]\n"
                       "2. Paste here: ")
    if not os.path.isfile(input_path):
        # If Path is a Folder
        folder_contents = [file for file in os.listdir(input_path) if file.endswith(".wav")]
    else:
        # If Path is a File
        folder_contents = [os.path.basename(input_path)]
        input_path = input_path[:-len(folder_contents[0])]

    return input_path, folder_contents


def make_mono(x):
    if len(x.shape) > 1:
        x_mono = 0.5 * np.sum(x, axis=1)
    else:
        x_mono = x
    return x_mono


def compute_sc_lin(data, fs):
    # Get Compy of audio data
    x = np.array(data, copy=True)
    # Make Mono if Stereo
    x = make_mono(x)
    # Compute N
    N = x.size
    # Compute FFT
    mX = np.abs(np.fft.rfft(x))
    # Compute Linear Spectral Centroid
    mX_weighted = np.sum([mX[i] * (i / N * fs) for i in range(mX.size)])
    # Devide Weighted Magnitudes by Sum of Magnitudes to derive weight
    sc_linear = mX_weighted / np.sum(mX, axis=0)
    # calculate Percentage of Brightness
    sc_lin_percent = sc_linear / (fs/2) * 100
    # Return spectral centroid linear
    return sc_linear, sc_lin_percent


# Apply Gain Offset using STFT
def apply_stft_eq(x, fs, centerfreq, gain):
    # Define FFF Size
    M = 8193*16
    N = int(2**np.ceil(np.log2(M*8)))
    # Derive CenterFrequency Index
    centerfreq_bin = int(centerfreq/fs*N)
    # Derive linear Gain Values
    gain_lin = 10 ** (gain/20)
    x_applied_gain = []
    for ch in range(x.shape[1]):
        # Perform STFT
        frequencies, time, Zxx = stft(x[:, ch], fs, nperseg=M, nfft=N)
        # Apply Gain Factors
        Zxx_applied_gain = np.zeros_like(Zxx)
        Zxx_applied_gain[centerfreq_bin:, :] = Zxx[centerfreq_bin:, :] * gain_lin
        Zxx_applied_gain[:centerfreq_bin, :] = Zxx[:centerfreq_bin, :] / gain_lin
        # Perform ISTFT
        _, x_channel_gained = istft(Zxx_applied_gain, fs, nperseg=M, nfft=N)
        # Collect ISFT for each channel
        x_applied_gain.append(x_channel_gained)
    # Transpose Channels to [samples, channels]
    x_applied_gain = np.array(x_applied_gain).transpose()
    return x_applied_gain


'''
-------------------------------------- MAIN PROCESS --------------------------------------
'''


def main():
    # Get Path
    path, files = get_base_path()
    for file in files:
        # Print Current File
        print(f"File: {file[:-4]}")
        # Load File
        x, fs = sf.read(f"{path}/{file}")
        # Compute Spectral Centroid Lin
        sc_lin_pre, _ = compute_sc_lin(x, fs)
        print(f"Spectral Centroid (Linear) Preprocessing: {sc_lin_pre}hz")

        # Weightings for Rebalancing Magnitudes (0 has to be first element)
        amplifications_dB = [0, -4, -3, -2., -1., -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1., 2., 3, 4]
        # Apply Various Gains
        for g in amplifications_dB:
            # Filter Audio Signal
            x_post_eq = apply_stft_eq(x, fs, sc_lin_pre, g)
            # Remeasure SC after Adjustment
            sc_lin_post, _ = compute_sc_lin(x_post_eq, fs)
            # Set SC Reference to SC Val at 0 gain adjustment
            if g == 0:
                sc_lin_pre = sc_lin_post

            # Measure SC Delta
            sc_delta = sc_lin_post-sc_lin_pre
            sc_delta = f"+{sc_delta:.0f}" if sc_delta >= 0 else f"{sc_delta:.0f}"
            # Normalise Loudness (to -3dB)
            out_data = (x_post_eq / np.max(np.abs(x_post_eq))) * 0.707
            # Define Output Name
            g_str = f"+{g:.3f}" if g >= 0 else f"{g:.2f}"
            out_name = f"{file[:-4]}{g_str}dB_sc{sc_lin_post:.0f}hz_scDelta{sc_delta}hz.wav"
            out_dir = f"{path}/{file[:-4]}"

            # Make Output Folder if not existant
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            # Define Output Path
            out_path = f"{out_dir}/{out_name}"
            # Write File
            sf.write(out_path, out_data, samplerate=fs)

            # Print Results
            print(f"Spectral Centroid (Linear) PostProcessing: {sc_lin_post:.0f}hz")
            print(f"Delta in Spectral Centroid: {sc_delta}hz")


# Main Function
if __name__ == "__main__":
    main()
