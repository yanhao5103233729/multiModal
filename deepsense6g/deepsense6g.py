#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @time: Sep 25 2023
# @software: Spyder
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import scipy.io as scipyio
import matplotlib.pyplot as plt

from tqdm import tqdm

# Absolute path of the folder containing the units' folders and scenarioX.csv
scenario_folder = r'/Users/yanhao/scenario9_dev/'

#%%

# Fetch scenario CSV
try:
    csv_file = [f for f in os.listdir(scenario_folder) if f.endswith('csv')][0]
    csv_path = os.path.join(scenario_folder, csv_file)
except:
    raise Exception(f'No csv file inside {scenario_folder}.')

# Load CSV to dataframe
dataframe = pd.read_csv(csv_path)
print(f'Columns: {dataframe.columns.values}')
print(f'Number of Rows: {dataframe.shape[0]}')

#%%

N_BEAMS = 64
n_samples = 100
pwr_rel_paths = dataframe['unit1_pwr_60ghz'].values
pwrs_array = np.zeros((n_samples, N_BEAMS))

for sample_idx in tqdm(range(n_samples)):
    pwr_abs_path = os.path.join(scenario_folder,
                                pwr_rel_paths[sample_idx])
    pwrs_array[sample_idx] = np.loadtxt(pwr_abs_path)

#%%

# Select specific samples to display
selected_samples = [5, 10, 20]
beam_idxs = np.arange(N_BEAMS) + 1
plt.figure(figsize=(10,6))
plt.plot(beam_idxs, pwrs_array[selected_samples].T)
plt.legend([f'Sample {i}' for i in selected_samples])
plt.xlabel('Beam indices')
plt.ylabel('Power')
plt.grid()

#%%

img_rel_paths = dataframe['unit1_rgb'].values
fig, axs = plt.subplots(figsize=(10,4), ncols=len(selected_samples), tight_layout=True)
for i, sample_idx in enumerate(selected_samples):
    img_abs_path = os.path.join(scenario_folder, img_rel_paths[sample_idx])
    img = plt.imread(img_abs_path)
    axs[i].imshow(img)
    axs[i].set_title(f'Sample {sample_idx}')
    axs[i].get_yaxis().set_visible(False)
    axs[i].get_xaxis().set_visible(False) 

#%%

# BS position (take the first because it is static)
bs_pos_rel_path = dataframe['unit1_loc'].values[0]
bs_pos = np.loadtxt(os.path.join(scenario_folder,
                                 bs_pos_rel_path))
# UE positions
pos_rel_paths = dataframe['unit2_loc_cal'].values
pos_array = np.zeros((n_samples, 2)) # 2 = lat & lon

# Load each individual txt file
for sample in range(n_samples):
    pos_abs_path = os.path.join(scenario_folder,
                                pos_rel_paths[sample])
    pos_array[sample] = np.loadtxt(pos_abs_path)

# Prepare plot: We plot on top of a Google Earth screenshot
gps_img = plt.imread(scenario_folder + 'resources/scen9_gps_img.png')

# Function to transform coordinates
def deg_to_dec(d, m, s, direction='N'):
    if direction in ['N', 'E']:
        mult = 1
    elif direction in ['S', 'W']:
        mult = -1
    else:
        raise Exception('Invalid direction.')

    return mult * (d + m/60 + s/3600)

# GPS coordinates from the bottom left and top right coordinates of the screenshot
gps_bottom_left, gps_top_right = ((deg_to_dec(33, 25, 8.94, 'N'),
                                   deg_to_dec(111, 55, 44.82, 'W')),
                                  (deg_to_dec(33, 25, 10.25, 'N'),
                                   deg_to_dec(111, 55, 43.50, 'W')))

best_beams = np.argmax(pwrs_array, axis=1)
fig, ax = plt.subplots(figsize=(6,8), dpi=100)
ax.imshow(gps_img, aspect='auto', zorder=0,
          extent=[gps_bottom_left[1], gps_top_right[1],
                  gps_bottom_left[0], gps_top_right[0]])

scat = ax.scatter(pos_array[selected_samples,1], pos_array[selected_samples,0], edgecolor='black', lw=0.7,
                  c=(best_beams[selected_samples] / N_BEAMS), vmin=0, vmax=1,
                  cmap=matplotlib.colormaps['jet'])

cbar = plt.colorbar(scat)
cbar.set_ticks([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
cbar.ax.set_yticklabels(['1', '8', '16', '24', '32', '40', '48', '56', '64'])
cbar.ax.set_ylabel('Best Beam Index', rotation=-90, labelpad=10)
ax.scatter(bs_pos[1], bs_pos[0], s=100, marker='X', color='red', label='BS')
ax.legend()
ax.ticklabel_format(useOffset=False, style='plain')
ax.tick_params(axis='x', labelrotation=45)
ax.set_xlabel('Longitude [º]')
ax.set_ylabel('Latitude [º]')

# We see about 2.5 car passes.

#%%

lidar_rel_paths =  dataframe['unit1_lidar_SCR'].values

# Compare noisy with denoised.
lidar_sample_size = 216

# Using the first 40 samples (almost the complete first car pass)
n_samp_first_seq = 40

# Append the lidar samples to array to show across a pass
lidar_frame = np.zeros((n_samp_first_seq, lidar_sample_size))
for sample_idx in range(n_samp_first_seq):
    lidar_file = os.path.join(scenario_folder, lidar_rel_paths[sample_idx])
    lidar_frame[sample_idx] = scipyio.loadmat(lidar_file)['data'][:,0]

angle_lims = [-90,90]
sample_lims = [0,n_samp_first_seq]
plt.figure(figsize=(6,2), dpi=120)
plt.imshow(np.fliplr(np.flipud(lidar_frame)),
           extent=[angle_lims[0], angle_lims[1], sample_lims[0], sample_lims[1]],
           aspect='equal')

plt.xlabel('Angle [º]')
plt.ylabel('Sample index')

#%%

sample_idx = 20
radar_rel_paths = dataframe['unit1_radar'].values
radar_data = scipyio.loadmat(os.path.join(scenario_folder, radar_rel_paths[sample_idx]))['data']

RADAR_PARAMS = {'chirps':            128, # number of chirps per frame
                'tx':                  1, # transmitter antenna elements
                'rx':                  4, # receiver antenna elements
                'samples':           256, # number of samples per chirp
                'adc_sampling':      5e6, # Sampling rate [Hz]
                'chirp_slope': 15.015e12, # Ramp (freq. sweep) slope [Hz/s]
                'start_freq':       77e9, # [Hz]
                'idle_time':           5, # Pause between ramps [us]
                'ramp_end_time':      60} # Ramp duration [us]

samples_per_chirp = RADAR_PARAMS['samples']
n_chirps_per_frame = RADAR_PARAMS['chirps']
C = 3e8
chirp_period = (RADAR_PARAMS['ramp_end_time'] + RADAR_PARAMS['idle_time']) * 1e-6

RANGE_RES = ((C * RADAR_PARAMS['adc_sampling']) /
                    (2*RADAR_PARAMS['samples'] * RADAR_PARAMS['chirp_slope']))

VEL_RES_KMPH = 3.6 * C / (2 * RADAR_PARAMS['start_freq'] *
                          chirp_period * RADAR_PARAMS['chirps'])

min_range_to_plot = 5
max_range_to_plot = 15 # m
# set range variables
acquired_range = samples_per_chirp * RANGE_RES
first_range_sample = np.ceil(samples_per_chirp * min_range_to_plot /
                            acquired_range).astype(int)
last_range_sample = np.ceil(samples_per_chirp * max_range_to_plot /
                            acquired_range).astype(int)
round_min_range = first_range_sample / samples_per_chirp * acquired_range
round_max_range = last_range_sample / samples_per_chirp * acquired_range

# Range-Velocity Plot
vel = VEL_RES_KMPH * n_chirps_per_frame/2
ang_lim = 75 # comes from array dimensions and frequencies

def minmax(arr):
    return (arr - arr.min())/ (arr.max()-arr.min())

def range_velocity_map(data):
    data = np.fft.fft(data, axis=1) # Range FFT
    # data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, axis=2) # Velocity FFT
    data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis = 0) # Sum over antennas
    data = np.log(1+data)
    return data

def range_angle_map(data, fft_size = 64):
    data = np.fft.fft(data, axis = 1) # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis = 0) # Angle FFT
    data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis = 2) # Sum over velocity
    return data.T

fig, axs = plt.subplots(figsize=(8,6), ncols=2, tight_layout=True)

# # Range-Angle Plot
radar_range_ang_data = range_angle_map(radar_data)[first_range_sample:last_range_sample]
axs[0].imshow(minmax(radar_range_ang_data), aspect='auto',
              extent=[-ang_lim, +ang_lim, round_min_range, round_max_range],
              cmap='seismic', origin='lower')
axs[0].set_xlabel('Angle [$\degree$]')
axs[0].set_ylabel('Range [m]')


radar_range_vel_data = range_velocity_map(radar_data)[first_range_sample:last_range_sample]
axs[1].imshow(minmax(radar_range_vel_data), aspect='auto',
              extent=[-vel, +vel, round_min_range, round_max_range],
              cmap='seismic', origin='lower')
axs[1].set_xlabel('Velocity [km/h]')
axs[1].set_ylabel('Range [m]')

#%%
    
import open3d as o3d
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np

# Path to lidar (pointcould) files
path = r'/Users/yanhao/scenario31/unit1/lidar_data/'
address = natsorted(os.listdir(path))
outpath = '.'
vis = o3d.visualization.Visualizer()

vis.create_window(visible=True)
vis.poll_events()
vis.update_renderer()
cloud = o3d.io.read_point_cloud(path+"/"+ address[0])
vis.add_geometry(cloud)

# (optional) In case a certain view is desired for the matplotlib shots:
params = {
    "field_of_view" : 60.0,
    "front" : [ -0.01093, 0.0308, 0.9994 ],
    "lookat" : [ -18.9122, -18.4687, 7.3131],
    "up" : [ 0.5496, 0.8351, -0.0197 ],
    "zoom" : 0.3200
    } # These parameters can be copied directly from the visualizer by doing Ctrl+C

opt = vis.get_render_option()
o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), params['zoom'])
o3d.visualization.ViewControl.set_lookat(vis.get_view_control(), params['lookat'])
o3d.visualization.ViewControl.set_front(vis.get_view_control(),params['front'])
o3d.visualization.ViewControl.set_up(vis.get_view_control(), params['up'])

for i in range(1): # number of frames to load
    cloud.points = o3d.io.read_point_cloud(path+"/"+ address[i]).points
    print(path+"/"+ address[i])
    vis.update_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()

    ## Plot with matplotlib
    color = np.asarray(vis.capture_screen_float_buffer(True))
    plt.imshow(color)
    plt.title(f'Sample {i+1}')
    plt.savefig(outpath+"/"+ address[i].split('.')[0] + '.png', dpi=300)
    plt.tight_layout()
    plt.pause(0.1)
vis.destroy_window()

#%%

import open3d as o3d
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np

# Path to lidar (point cloud) files
path = r'/Users/yanhao/scenario31/unit1/lidar_data/'
address = natsorted(os.listdir(path))
outpath = '.'
vis = o3d.visualization.Visualizer()

vis.create_window(visible=True)
vis.poll_events()
vis.update_renderer()
cloud = o3d.io.read_point_cloud(path+"/"+ address[0])
vis.add_geometry(cloud)

# (optional) In case a certain view is desired for the matplotlib shots:
params = {
    "field_of_view": 60.0,
    "front": [-0.01093, 0.0308, 0.9994],
    "lookat": [-18.9122, -18.4687, 7.3131],
    "up": [0.5496, 0.8351, -0.0197],
    "zoom": 0.3200
    } # These parameters can be copied directly from the visualizer by doing Ctrl+C

opt = vis.get_render_option()
vis.get_view_control().set_front(params['front'])
vis.get_view_control().set_lookat(params['lookat'])
vis.get_view_control().set_up(params['up'])
vis.get_view_control().set_zoom(params['zoom'])

for file_name in address:
    if file_name.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(os.path.join(path, file_name))
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

vis.destroy_window()

{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 179.38339215148096, 112.48549607166876, 73.042827130073476 ],
			"boundingbox_min" : [ -198.82097073535132, -170.11726769524338, -58.823731442776136 ],
			"field_of_view" : 60.0,
			"front" : [ 0.0, 0.0, 1.0 ],
			"lookat" : [ -9.718789291935181, -28.815885811787311, 7.1095478436486701 ],
			"up" : [ 0.0, 1.0, 0.0 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
