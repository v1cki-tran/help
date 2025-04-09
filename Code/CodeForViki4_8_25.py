# -*- coding: utf-8 -*-


import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime


from cloud_motion import cloud_motion
from cloud_detection import cloud_detection


now_list = []
k=0
a = datetime.datetime(2025,3,28,11,34,59)
while k<11:
    b = a + datetime.timedelta(seconds=30)
    now_list.append(b)
    a=b
    k=k+1


full_array=[]

folder = os.listdir('set3')
seg_frame1,__,____,_____=cloud_detection('set3/2025-03-22_22-35-03.jpg')



j = 0
for images in folder:
    image_filename='set3'+'/'+ images
    csv_array=[]
    seg_frame2, percent_cover, file_name, motion_mask = cloud_detection(image_filename)
    csv_array=cloud_motion(seg_frame1, seg_frame2, percent_cover, image_filename, 
                 now_list[j], csv_array, motion_mask)
    seg_frame1=seg_frame2
    full_array.append(csv_array[0])
    j=j+1
    print("dickskucekr")
    if np.isnan(csv_array[0][3]):
        print("NaN alert!!")
        break
    
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
plt.locator_params(axis='x', nbins=5)
plt.xlabel("Time")
plt.ylabel("Percent coverage (%)")
plt.grid(True)
plt.savefig("Set3_Plot1.png")

plt.figure(figsize=(10,12))
plt.plot(listical[0], listical[3])
plt.title('Average magnitude')
plt.tick_params(pad=10)
plt.xticks(fontsize=16, rotation=65)
plt.yticks(fontsize=14)
plt.locator_params(axis='x', nbins=5)
plt.xlabel("Time")
plt.ylabel("Average Magnitude")
plt.grid(True)
plt.savefig("Set3_Plot2.png")




