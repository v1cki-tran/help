import cv2
import numpy as np
from datetime import datetime

from cloud_detection import cloud_detection

def cloud_motion(seg_frame1, seg_frame2, percent_cover, 
                 image_filename, now, csv_array, motion_mask):

    seg_frame1 = cv2.bitwise_and(seg_frame1, seg_frame1, mask=motion_mask)
    seg_frame2 = cv2.bitwise_and(seg_frame2, seg_frame2, mask=motion_mask)
    
    int(percent_cover)
    
    print("pre thresh Sum Seg1:", np.sum(seg_frame1))
    print("pre thresh Sum Seg2:", np.sum(seg_frame2))
    
    # Ensure binary masks (clouds = 255, background = 0)
    _, seg_frame1 = cv2.threshold(seg_frame1, 127, 255, cv2.THRESH_BINARY)
    _, seg_frame2 = cv2.threshold(seg_frame2, 127, 255, cv2.THRESH_BINARY)
    flow = cv2.calcOpticalFlowFarneback(seg_frame1, seg_frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    print("Sum Seg1:", np.sum(seg_frame1))
    print("Sum Seg2:", np.sum(seg_frame2))
    print("Sum flow:", np.sum(flow))

    # Calculate magnitude and angle of flow vectors
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=True)
    
    cloud_mask = seg_frame1 > 0
    avg_magnitude = np.mean(magnitude[cloud_mask])
    print("summed mag:", np.sum(magnitude[cloud_mask]))
    print("average mag:", avg_magnitude)
    avg_angle = np.mean(angle[cloud_mask])
    print("average angle:", avg_angle)

    # Determine the cardinal direction (currently abaritrary numbers) 
    if 247.5 <= avg_angle < 292.5:
        direction = "North"
    elif 292.51 <= avg_angle < 337.5:
        direction = "North-East"
    elif 337.51 <= avg_angle < 22.5:
        direction = "East"
    elif 22.51 <= avg_angle < 67.5:
        direction = "South-East"
    elif 67.51 <= avg_angle < 112.5:
        direction = "South"
    elif 112.51 <= avg_angle < 157.5:
        direction = "South-West"
    elif 157.51 <= avg_angle < 202.5:
        direction = "West"
    else:
        direction = "North-West"
        
    image_list = image_filename.split("/")
    image_name = image_list[-1]
    csv_array.append([now.strftime('%Y-%m-%d %H:%M:%S'), image_name, percent_cover, avg_magnitude, direction])
    return csv_array


seg1 = cv2.imread('capstonestuff/2025-03-23_18-33-45.jpg', cv2.IMREAD_GRAYSCALE)
image_filename = 'capstonestuff/2025-03-23_18-34-15.jpg'
percent = 10
#filename = 'capstonestuff/2025-03-23_18-33-45.jpg'
now = datetime.now()
csv_array = []
height,width = seg1.shape



#seg2, percent_cover, file_name, motion_mask = cloud_detection(image_filename)

#csv_array = cloud_motion(seg1,seg2, percent_cover,file_name,now,csv_array,motion_mask)
#print(csv_array)