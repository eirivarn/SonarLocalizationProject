import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Load data from CSV files
fft_data = pd.read_csv("../output/relative_pose_fft.csv")
tru_depth_data = pd.read_csv("../output/relative_pose_tru_depth.csv")

#sort by time
fft_data = fft_data.sort_values(by='time')
tru_depth_data = tru_depth_data.sort_values(by='time')

# Extract relevant columns (adjust column names as per your CSV structure)
fft_time = fft_data['time'].values
fft_distance = fft_data['distance'].values
fft_pitch = fft_data['pitch'].values
fft_heading = fft_data['heading'].values

tru_depth_time = tru_depth_data['time'].values
tru_depth_distance = tru_depth_data['distance'].values
tru_depth_pitch = tru_depth_data['pitch'].values
tru_depth_heading = tru_depth_data['heading'].values


# ----------- PLOT 1: Relative Net Distance Over Time -----------
plt.figure(figsize=(20, 6))
plt.xlabel('Time [s]')
plt.ylabel('Distance [cm]')
plt.grid(True)
plt.title('Relative Net Distance Over Time')
# plt.ylim([0, 400])
# plt.xlim([20, 140])

# Set the shift amount for time
bag_start_time = 1747996885.44

# Adjust FFT times relative to the minimum timestamp
time_shifted_fft = fft_time - bag_start_time
filtered_fft_distance = np.convolve(fft_distance, np.ones(5)/5, mode='same')
plt.plot(time_shifted_fft, filtered_fft_distance, label='FFT', color='red')
# plt.plot(time_shifted_fft, fft_distance, label='FFT Raw', color='blue')

# Adjust TRU-Depth times relative to the minimum timestamp
time_shifted_tru = tru_depth_time - bag_start_time
filtered_tru_distance = np.convolve(tru_depth_distance, np.ones(5)/5, mode='same')
plt.plot(time_shifted_tru, filtered_tru_distance, label='TRU-Depth', color='green')
# plt.plot(time_shifted_tru, tru_depth_distance, label='TRU-Depth Raw', color='orange')

plt.legend(loc='upper right')
plt.savefig('relative_net_distance_plot.png')
plt.close()

# ----------- PLOT 2: Pitch and Heading Over Time (FFT + TruDepth) -----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'bottom': 0.06, 'top': 0.945, 'hspace': 0.18, 'right': 0.95, 'left': 0.07})

#set y limits
# ax1.set_ylim([-45, 35])
# ax2.set_ylim([-40, 50])
# ax1.set_xlim([20, 145])
# ax2.set_xlim([20, 145])

# FFT Pitch and Heading
filtered_fft_pitch = np.convolve(np.degrees(fft_pitch), np.ones(5)/5, mode='same')
filtered_fft_heading = np.convolve(np.degrees(fft_heading), np.ones(5)/5, mode='same')
ax1.plot(time_shifted_fft, filtered_fft_pitch, label='FFT Pitch', color='purple')
# ax1.plot(time_shifted_fft, np.degrees(fft_pitch), label='FFT Raw Pitch', color='red')

ax2.plot(time_shifted_fft, filtered_fft_heading, label='FFT Heading', color='purple')
# ax2.plot(time_shifted_fft, np.degrees(fft_heading), label='FFT Raw Heading', color='red')


# TRU-Depth Pitch and Heading
filtered_tru_pitch = np.convolve(np.degrees(tru_depth_pitch), np.ones(5)/5, mode='same')
filtered_tru_heading = np.convolve(np.degrees(tru_depth_heading), np.ones(5)/5, mode='same')
ax1.plot(time_shifted_tru, filtered_tru_pitch, label='Tru-Depth Pitch', color='orange')
# ax1.plot(time_shifted_tru, np.degrees(tru_depth_pitch), label='Tru-Depth Raw Pitch', color='blue')

ax2.plot(time_shifted_tru, filtered_tru_heading, label='Tru-Depth Heading', color='orange')
# ax2.plot(time_shifted_tru, np.degrees(tru_depth_heading), label='Tru-Depth Raw Heading', color='blue')

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Net Pitch [deg]')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Net Heading [deg]')

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

ax1.grid(True)
ax2.grid(True)

fig.suptitle("Net Orientation Estimates")
plt.savefig('pitch_heading_plot.png')
plt.close()