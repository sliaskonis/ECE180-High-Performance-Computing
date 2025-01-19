import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# colors_list = (['red', 'blue'], ['green', 'orange'], ['purple', 'yellow'])
colors_list = (["#e60049", "#0bb4ff"], ["#50e991", "#e6d800"], ["#9b19f5", "#ffa300"], ["#dc0ab4", "#b3d4ff"])#, "#00bfa0"]
plt.figure(figsize=(10, 6))
# Load the data
# files = os.listdir('results_csv')

files = ['conv2d_2D_2D.csv', 'conv2d_pad.csv']#'con2d_2D_2D_doubles.csv'] #'conv2d_pad.csv']

labels_list = (['CPU Execution Times', 'GPU Execution Times'], ['CPU Execution Times with Doubles', 'GPU Execution Times with Doubles'], ['CPU Execution Times with Padding', 'GPU Execution Times with Padding'])


for colors, file, labels in zip(colors_list, files, labels_list):
    if file.endswith('.csv'):
        data = pd.read_csv('results_csv/' + file)
        # Create the plot
        # plt.figure(size=(10, 6))
        plt.plot(np.array(data['sizes']), np.array(data['cpu_means']), label=labels[0], color=colors[0])
        # plt.fill_between(data['sizes'], np.array(data['cpu_means']) - np.array(data['cpu_stds']), np.array(data['cpu_means']) + np.array(data['cpu_stds']), alpha=0.2, color=colors[0])
        plt.scatter(np.array(data['sizes']), np.array(data['cpu_means']), color=colors[0])
        
        plt.plot(np.array(data['sizes']), np.array(data['gpu_means']), label=labels[1], color=colors[1])
        # plt.fill_between(data['sizes'], np.array(data['gpu_means']) - np.array(data['gpu_stds']), np.array(data['gpu_means']) + np.array(data['cpu_stds']), alpha=0.2, color=colors[1])
        plt.scatter(np.array(data['sizes']), np.array(data['gpu_means']), color=colors[1])
        
        plt.legend()
        plt.xlabel('Input Image Size')
        plt.ylabel('Time (ms)')
        plt.grid()

plt.show()
