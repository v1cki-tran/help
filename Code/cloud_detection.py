import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from skimage.feature import local_binary_pattern
from skimage import data, img_as_float
from skimage import exposure

def cloud_detection(file_name: str, debug=False):

    image = cv2.imread(file_name)
    #local contrast adjustment filter
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ##################################################################################
    """Y Method
    thresh = cv2.adaptiveThreshold(y_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 30)
    could be good for object detection/removal"""
    #read image and check
    img = cv2.imread(file_name)
    if img is None:
        print(f"Error with image: {image}")
    
    #convert to YCC for Y-channel adaptive threshold 
    ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr_img[:, :, 0]
    cb_channel = ycbcr_img[:, :, 1]
    cr_channel = ycbcr_img[:, :, 2]
    y_channel_eq = clahe.apply(y_channel)
    
    #split channels and equalize
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel_eq = clahe.apply(red_channel)
    green_channel_eq = clahe.apply(green_channel)
    blue_channel_eq = clahe.apply(blue_channel)
    
    #get image dimensions
    height, width = y_channel.shape
    total_pixels = height * width
    
    
    #create mask for image circle
    black_mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = 1500
    cv2.circle(black_mask, center, radius, 255, -1)

    motion_mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = 1000
    cv2.circle(motion_mask, center, radius, 255, -1)
    
    #calculate total pixels + debuging print statments
    total_pixels = total_pixels - (np.sum(black_mask)/255)
    #print('image pixels: ',(width * height))
    #print("black mask: ", np.sum(black_mask))
    #print("total pixels no mask: ", total_pixels)
    
    
    #threshold values 250+ and check if greater than 100
    _, sun_mask = cv2.threshold(y_channel, 250, 255, cv2.THRESH_BINARY)
    if (np.sum(sun_mask) / 255) > 100:
        #get image moment for x and y
        moment = cv2.moments(sun_mask)
        if moment["m00"] != 0:
            sun_x = int(moment["m10"] / moment["m00"])
            sun_y = int(moment["m01"] / moment["m00"])

        #create sun mask
        sun_detection = np.zeros_like(sun_mask, dtype=np.uint8)
        size = 300
        cv2.circle(sun_detection, (sun_x, sun_y), size, 255, -1)
    
        sun_pixels = (np.sum(sun_mask) / 255)
        useful_pixels = total_pixels - sun_pixels
        #print("useful pixels: ", useful_pixels)
    
    else:
        #return default values
        useful_pixels = total_pixels
        sun_detection = np.zeros_like(sun_mask, dtype=np.uint8)
    
    ##################################################################################
    ##################################################################################
    """Y-channel adaptive threshold"""
    
    #threshold the Y channel and use otsu to find optimal threshold value 
    otsu_thresh, _ = cv2.threshold(y_channel_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = max(otsu_thresh + 10, 0)  #increase threshold by 10 
    _, thresh = cv2.threshold(y_channel_eq, adjusted_thresh, 255, cv2.THRESH_BINARY)
    binary_image_1 = cv2.subtract(thresh, sun_detection)
    binary_image_1 = cv2.bitwise_and(binary_image_1, binary_image_1, mask=black_mask)
    
    ##################################################################################
    ##################################################################################
    """GB green blue ratio"""
    
    #threshold the diffrence between green and blue channel
    gb_image = cv2.divide(green_channel_eq.astype(np.float32), blue_channel_eq.astype(np.float32))
    _, binary_image_2 = cv2.threshold(gb_image, 0.79, 255, cv2.THRESH_BINARY) #threshold at 0.79
    binary_image_2 = np.clip(binary_image_2 - sun_detection, 0, 255)
    binary_image_2 = cv2.bitwise_and(binary_image_2, binary_image_2, mask=black_mask)
    
    ##################################################################################
    ##################################################################################
    """RB red blue ratio"""
    
    #threshold the diffrence between red and blue channel
    rb_image = cv2.divide(red_channel_eq.astype(np.float32), blue_channel_eq.astype(np.float32))
    _, binary_image_3 = cv2.threshold(rb_image, 0.68, 255, cv2.THRESH_BINARY) #threshold at 0.68
    binary_image_3 = np.clip(binary_image_3 - sun_detection, 0, 255)
    binary_image_3 = cv2.bitwise_and(binary_image_3, binary_image_3, mask=black_mask)
    
    ##################################################################################
    ##################################################################################
    """RG red green ratio"""
    
    #threshold the diffrence between red and green channel
    rg_image = cv2.divide(red_channel_eq.astype(np.float32), green_channel_eq.astype(np.float32))
    _, binary_image_4 = cv2.threshold(rg_image, 0.9, 255, cv2.THRESH_BINARY) #Threshold at 0.9
    binary_image_4 = np.clip(binary_image_4 - sun_detection, 0, 255)
    binary_image_4 = cv2.bitwise_and(binary_image_4, binary_image_4, mask=black_mask)
    
    ##################################################################################
    ##################################################################################
    """Binary pattern """

    #create binary pattern
    radius = 1
    n_points = int(8 * radius)
    lbp = local_binary_pattern(green_channel_eq, n_points, radius, method='uniform')
    
    #normalize binary pattern and scale 0-255
    ldp_255 = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(ldp_255, (61,61), 0) 
    
    #threshol the blured image
    _, binary_image_5 = cv2.threshold(blurred, 145, 255, cv2.THRESH_BINARY) #(145, 255)
    binary_image_5 = np.clip(binary_image_5 - sun_detection, 0, 255)
    binary_image_5 = cv2.bitwise_and(binary_image_5, binary_image_5, mask=black_mask)
    
    ##################################################################################
    ##################################################################################
    #create a mask where at least 4 out of 5 masks agree
    agreement_threshold = 4  #variable for agreement threshold
    combined_agree_mask = np.where(((binary_image_1 / 255) + (binary_image_2 / 255) + (binary_image_3 / 255) + (binary_image_4 / 255) + (binary_image_5 / 255)) >= agreement_threshold, 255, 0).astype(np.uint8) 
    cloud_pixels = np.sum(combined_agree_mask)/255
    percent_cover = round((cloud_pixels / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    
    #plt.imshow(combined_agree_mask, cmap='gray')

    if debug:
        print("cloud pixels:", cloud_pixels)
        print("useful pixels:", useful_pixels)
        print("Percent cover: ", percent_cover)
    
    
    """
    #test each method and mean 
    percent_cover_y = round(((np.sum(binary_image_1)/255) / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    percent_cover_gb = round(((np.sum(binary_image_2)/255) / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    percent_cover_rb = round(((np.sum(binary_image_3)/255) / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    percent_cover_rg = round(((np.sum(binary_image_4)/255) / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    percent_cover_binary = round(((np.sum(binary_image_5)/255) / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    if useful_pixels > 0:
        mean = round((20 / (255.0 * useful_pixels)) * (np.sum(binary_image_1) + np.sum(binary_image_2) + np.sum(binary_image_3) + np.sum(binary_image_4) + np.sum(binary_image_5)), 2)
    else:
        mean = 0
    
    print('Mean cloud _cover is =', mean, '%')
    print(percent_cover_y)
    """
    
    ##################################################################################
    ##################################################################################
    # Plot original image
    fov = 164
    pfov = 120
    height, width = y_channel.shape
    
    # Image center
    xc = width // 2
    yc = height // 2
    dim = np.sqrt(width ** 2.0 + height ** 2.0) #diagonal
    
    # meshgrid for pixel locations
    i = np.arange(width)
    j = np.arange(height)
    i, j = np.meshgrid(i, j)
    
    # Original focal length
    ofoc = dim / (2 * np.tan(pfov * np.pi / 360))
    ofocinv = 1.0 / ofoc
    
    # Distance from center
    xd = i - xc
    yd = j - yc
    rd = np.hypot(xd, yd)
    
    # Apply fisheye transformation
    phi = np.arctan(ofocinv * rd)
    f = dim * 180 / (fov * np.pi)
    rr = f * phi
    
    if debug:
        print("rr:", rr)
    
    
    
    alpha = ofocinv
    rdmask = (rd > 0) 
    dAdA = np.zeros_like(rd, dtype=np.float32)
    drpdr = f * (alpha / (1.0 + (alpha * rd)**2))
    r_ratio = (rr[rdmask] / rd[rdmask]) * drpdr[rdmask]
    dAdA[rdmask] = r_ratio
    geom_weight_map = np.ones_like(rd, dtype=np.float32)
    geom_weight_map[rdmask] = 1.0 / dAdA[rdmask]
    
    if debug:
        print("rdmask shape:",rdmask.shape)
        print("geom shape:", geom_weight_map.shape)
    
    
    weighted_image = combined_agree_mask * geom_weight_map
    
    #plt.figure()
    #plt.imshow(geom_weight_map, cmap='gray')
    #plt.figure()
    #plt.imshow(combined_agree_mask, cmap='gray')
    #plt.figure()
    #plt.imshow(weighted_image, cmap='gray')
    
    
    weighted_image_formath = (weighted_image/ 255)
    
    
    top_divisor = np.sum(weighted_image_formath)
    if debug:
        print(top_divisor)
    
    #percent_cover_weighted = round((np.sum(weighted_image) / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    percent_cover_weighted = round((top_divisor / useful_pixels) * 100, 2) if useful_pixels > 0 else 0
    if debug:
        print("Weighted percent: ", percent_cover_weighted)
    

    return weighted_image, percent_cover_weighted, file_name, motion_mask
    
if __name__ == '__main__':
    cloud_detection('set3/2025-03-23_18-33-15.jpg')
    ##################################################################################
    ##################################################################################
"""
    plt.figure(dpi=150)
    plt.subplot(1, 3, 1)
    plt.xticks([]), plt.yticks([])
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Plot Combined Threshold
    plt.subplot(1, 3, 2)
    plt.xticks([]), plt.yticks([])
    plt.title("Combined Threshold")
    plt.imshow(combined_agree_mask, cmap='gray')
    
    # Plot Weighted Combined
    plt.subplot(1, 3, 3)
    plt.xticks([]), plt.yticks([])
    plt.title("Weighted Combined")
    plt.imshow(weighted_image, cmap='jet')
    
    #plt.savefig(str(image) + "_Plot1.png")
    plt.show()
    plt.close()
    
    #plt.figure(dpi=150)
    #plt.imshow(weighted_image, cmap='jet')
    #plt.colorbar()
    #plt.show()
    
    ##################################################################################
    ##################################################################################

    # Set up subplots with two rows and three columns
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
    axes = axes.ravel()
    
    # Define titles and images
    titles = ["Original", "Otsu", "GB", "RB", "RG", "Binary"]
    images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        binary_image_1,
        binary_image_2,
        binary_image_3,
        binary_image_4,
        binary_image_5,
        #combined_agree_mask,
    ]
    
    # Plot images in the subplots
    for i in range(6):
        axes[i].imshow(images[i], cmap='gray' if i > 0 else None)
        axes[i].set_title(titles[i])
        axes[i].axis("off")
    
    # Save and show the figure
    plt.tight_layout()
    #plt.savefig(str(image) + "_Plot2.png")
    plt.show()
    plt.close()
    
    #cv2.imshow("Combined", combined_agree_mask)
    #cv2.waitKey(0)
    
    print(percent_cover_weighted, file_name)
    

#cloud_detection('capstonestuff/2025-03-23_18-33-15.jpg')


"""
    
