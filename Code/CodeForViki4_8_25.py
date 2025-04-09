# -*- coding: utf-8 -*-


import cv2
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateFormatter
import pandas as pd
import numpy as np
import os
import datetime
import re

from cloud_motion import cloud_motion
from cloud_detection import cloud_detection

IMG_DIR = 'set5'

def datetime_from_filename(filename: str) -> datetime.datetime:
    return datetime.datetime(*[int(n) for n in re.split(r'[-_.]', filename)[:-1]])

folder = os.listdir(IMG_DIR)
a = datetime_from_filename(folder[0])

full_array=[]
seg_frame1,__,____,_____=cloud_detection(f'{IMG_DIR}/2025-03-22_22-50-03.jpg')
i = 0
for image_name in folder[1:]: # skip the first image since we grabbed it for the first seg_frame1
    image_path = IMG_DIR + '/' + image_name
    csv_array=[]
    seg_frame2, percent_cover, file_name, motion_mask = cloud_detection(image_path)
    csv_array=cloud_motion(seg_frame1, seg_frame2, percent_cover, image_path, 
                datetime_from_filename(image_name) , csv_array, motion_mask)
    seg_frame1=seg_frame2
    full_array.append(csv_array[0])
    i += 1
    print(f"\r{i}/{len(folder)} images processed ({round(100*i/len(folder))}%)...", end='')
    if np.isnan(csv_array[0][3]):
        print("NaN alert!!")
        break

print('\ndickskucekr', end='\r')

listical = [[],
            [],
            [],
            [],
            []]
for lines in full_array:
    n=0
    for terms in lines:
        listical[n].append(terms)
        n=n+1

    
plt.figure(figsize=(10,12))
plt.plot(listical[0], listical[2])
plt.title('Percent coverage(%)')
plt.tick_params(pad=10)
plt.xticks(fontsize=16, rotation=65)
plt.yticks(fontsize=14)
plt.ylim(0, 100)
plt.xlabel("Time")
plt.ylabel("Percent coverage (%)")
plt.grid(True)
plt.savefig(f"{IMG_DIR}_Plot1.png")

plt.figure(figsize=(10,12))
plt.plot(listical[0], listical[3])
plt.title('Average magnitude')
plt.tick_params(pad=10)
plt.xticks(fontsize=16, rotation=65)
plt.yticks(fontsize=14)
plt.xlabel("Time")
plt.ylabel("Average Magnitude")
plt.grid(True)
plt.savefig(f"{IMG_DIR}_Plot2.png")
