# Hank on Mar 8 2021 to find spatial coverage mismatch & other mismatch for pre_Collection & Collection-1 data
# import find_files
import os 
import numpy as np

import pathlib
data_dir = pathlib.Path("./")
cnn_file_list = list(data_dir.glob('demo*'))
# cnn_file_list = list(data_dir.glob('*.tif'))
cnn_file_list_str = np.array([str(f) for f in cnn_file_list])

# class All_Files:
    # mtl_cld_file_list = list()
    # cld_cld_file_list = list()
    # dnn_cld_file_list = list()
    # bqa_cld_file_list = list()
    # mtl_toa_file_list = list()
    # toa_toa_file_list = list()
    # ang_toa_file_list = list()


def find_list(DIR,is_L8=True,pattern=".TIF"):
    bqa_cld_file_list = list()
    
    # sceneID_DIR = CLDDIR
    mtl_n = 0
    for root, dirs, files in os.walk(DIR):
        for file in files:
            # print (file)
            if pattern in file:
                print(file)
                if (is_L8==True and 'LC08' in file) or (is_L8==False):
                    bqa_cld_file_list.append(os.path.join(root, file))
                
    return bqa_cld_file_list
  

def find_qa_list(DIR,is_L8=True,pattern=".TIF"):
    bqa_cld_file_list = list()
    
    # sceneID_DIR = CLDDIR
    mtl_n = 0
    for root, dirs, files in os.walk(DIR):
        for file in files:
            # print (file)
            if '_QA_PIXEL.TIF' in file and pattern in file:
                print(file)
                if (is_L8==True and 'LC08' in file) or (is_L8==False):
                    bqa_cld_file_list.append(os.path.join(root, file))
                
    return bqa_cld_file_list
    # return mtl_cld_file_list, cld_cld_file_list, dnn_cld_file_list, bqa_cld_file_list, mtl_toa_file_list, toa_toa_file_list


def find_qa_list_years(DIR,is_L8=True,pattern=".TIF",starty=1984,endy=1990):
    bqa_cld_file_list = list()
    
    # sceneID_DIR = CLDDIR
    mtl_n = 0
    for yi in range(starty,endy):
        for root, dirs, files in os.walk(DIR):
            for file in files:
                # print (file)
                if '_QA_PIXEL.TIF' in file and pattern+str(yi) in file:
                    print(file)
                    if (is_L8==True and 'LC08' in file) or (is_L8==False):
                        bqa_cld_file_list.append(os.path.join(root, file))
                
    return bqa_cld_file_list
    # return mtl_cld_file_list, cld_cld_file_list, dnn_cld_file_list, bqa_cld_file_list, mtl_toa_file_list, toa_toa_file_list

# HLS # HLS.L30.T13TEF.2020033T173658.v2.0.Fmask.tif
# ARD # LC08_ARD_TILE_014034_20210912_01_T1_SR_B4.TIF
def find_L30_qa_list_years(DIR,is_L8=True,sensor = "L30",pattern=".Fmask.tif",tile_id="14TLT",starty=1984,endy=1990):
    bqa_cld_file_list = list()
    for yi in range(starty,endy):
        for root, dirs, files in os.walk(DIR):
            for file in files:
                # print (file)
                if tile_id in file and '.Fmask.tif' in file and "."+str(yi) in file and sensor in file:
                    print(file)
                    # if (is_L8==True and 'LC08' in file) or (is_L8==False):
                    bqa_cld_file_list.append(os.path.join(root, file))
    return bqa_cld_file_list

def find_S30_qa_list_years(DIR,is_L8=True,sensor = "S30",pattern=".Fmask.tif",tile_id="14TLT",starty=1984,endy=1990):
    bqa_cld_file_list = list()
    for yi in range(starty,endy):
        for root, dirs, files in os.walk(DIR):
            for file in files:
                # print (file)
                if tile_id in file and '.Fmask.tif' in file and "."+str(yi) in file and sensor in file:
                    print(file)
                    # if (is_L8==True and 'LC08' in file) or (is_L8==False):
                    bqa_cld_file_list.append(os.path.join(root, file))
    return bqa_cld_file_list

def find_CNN_mask(DIR,is_L8=True,pattern=".tif", pattern2=".tif"):
    bqa_cld_file_list = list()
    
    # sceneID_DIR = CLDDIR
    mtl_n = 0
    for root, dirs, files in os.walk(DIR):
        for file in files:
            if ((is_L8==True and 'LC08' in file) or (is_L8==False)) and pattern in file and pattern2 in file:                         #  and "_T1" in file '.CQAV14.1.tif' in file 
                print(file)
                if (is_L8==True and 'LC08' in file) or (is_L8==False):
                    bqa_cld_file_list.append(os.path.join(root, file))
                
    return bqa_cld_file_list
    # return mtl_cld_file_list, cld_cld_file_list, dnn_cld_file_list, bqa_cld_file_list, mtl_toa_file_list, toa_toa_file_list
def find_CS_mask(DIR,is_L8=True,pattern=".tif"):
    bqa_cld_file_list = list()
    
    # sceneID_DIR = CLDDIR
    mtl_n = 0
    for root, dirs, files in os.walk(DIR):
        for file in files:
            if '.CS.tif' in file and 'LC08' in file and pattern in file:                         #  and "_T1" in file
                print(file)
                if (is_L8==True and 'LC08' in file) or (is_L8==False):
                    bqa_cld_file_list.append(os.path.join(root, file))
                
    return bqa_cld_file_list