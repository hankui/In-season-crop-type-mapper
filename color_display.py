# d:\mycode.RT\mycode.research\ABI\
# revised on to use image as input
# Hank on Sep 11, 2020 
# import color_display

import numpy as np 
import rasterio

def color_display(input_file, dsr_file, output_tif="./output.tif"):
    ## ***************************
    ## interpret colour dsr 
    f = open(dsr_file, "r")
    i = 0
    xarray_list = list()
    for x in f:
        # print(x)
        i = i+1
        if i>=2:
            xarray = x.split( )
            xarray_list.append(xarray)
    
    n = len(xarray_list)
    left  = np.full([n], fill_value=0.0, dtype=np.float32) ## must use fill
    right = np.full([n], fill_value=0.0, dtype=np.float32) ## must use fill
    RR    = np.full([n], fill_value=0, dtype=np.uint8) ## must use fill
    GG    = np.full([n], fill_value=0, dtype=np.uint8) ## must use fill
    BB    = np.full([n], fill_value=0, dtype=np.uint8) ## must use fill
    
    for i in range(n):
        left [i] =  float(xarray_list[i][0])
        right[i] =  float(xarray_list[i][1])
        RR   [i] =  int  (xarray_list[i][2])
        GG   [i] =  int  (xarray_list[i][3])
        BB   [i] =  int  (xarray_list[i][4])
    
    ## ***************************
    ## interpret colour dsr 
    with rasterio.open(input_file) as dataset:
        naip_meta = dataset.profile
        image = dataset.read(1)
        
    out_image  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    # out_RR  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    # out_GG  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    # out_BB  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    for i in range(n):
        index = np.logical_and(image>=left[i], image<=right[i])
        out_image[0,index] = RR   [i]
        out_image[1,index] = GG   [i]
        out_image[2,index] = BB   [i]
        # if image[i,j]>=
    
    naip_meta['dtype'] = 'int8'
    naip_meta['count'] = 3
    naip_meta['nodata'] = 0
    # naip_meta['driver'] = "GTiff"
    with rasterio.open(output_tif, 'w', **naip_meta) as dst:
        dst.write(out_image.astype(np.int8))    



def color_display_from_image(image, dsr_file, output_tif="./output.tif"):
    ## ***************************
    ## interpret colour dsr 
    f = open(dsr_file, "r")
    i = 0
    xarray_list = list()
    for x in f:
        # print(x)
        i = i+1
        if i>=2:
            xarray = x.split( )
            xarray_list.append(xarray)
    
    n = len(xarray_list)
    left  = np.full([n], fill_value=0.0, dtype=np.float32) ## must use fill
    right = np.full([n], fill_value=0.0, dtype=np.float32) ## must use fill
    RR    = np.full([n], fill_value=0, dtype=np.uint8) ## must use fill
    GG    = np.full([n], fill_value=0, dtype=np.uint8) ## must use fill
    BB    = np.full([n], fill_value=0, dtype=np.uint8) ## must use fill
    
    for i in range(n):
        left [i] =  float(xarray_list[i][0])
        right[i] =  float(xarray_list[i][1])
        RR   [i] =  int  (xarray_list[i][2])
        GG   [i] =  int  (xarray_list[i][3])
        BB   [i] =  int  (xarray_list[i][4])
    
    ## ***************************
    ## interpret colour dsr 
    # with rasterio.open(input_file) as dataset:
        # naip_meta = dataset.profile
        # image = dataset.read(1)
        
    out_image  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    # out_RR  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    # out_GG  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    # out_BB  = np.full([3,image.shape[0], image.shape[1]], fill_value=0, dtype=np.uint8) ## must use fill
    for i in range(n):
        index = np.logical_and(image>=left[i], image<=right[i])
        out_image[0,index] = RR   [i]
        out_image[1,index] = GG   [i]
        out_image[2,index] = BB   [i]
        # if image[i,j]>=
    
    naip_meta = rasterio.profiles.DefaultGTiffProfile()
    # naip_meta['dtype'] = 'uint8'
    naip_meta['tiled'] = False
    naip_meta['count'] = 3
    # naip_meta['nodata'] = 0
    naip_meta['width'] = image.shape[1]
    naip_meta['height'] = image.shape[0]
    # naip_meta['driver'] = "GTiff"
    with rasterio.open(output_tif, 'w', **naip_meta) as dst:
        dst.write(out_image)    
        