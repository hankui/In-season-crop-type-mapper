# d:\mycode.RT\mycode.research\Landsat_8_BRDF\
# Hank on May 30, 2024 
# Hank on Aug 02, 2020 
# Hank on Dec 01, 2021 

# import Landsat_ARD_io

## *** ! in csv files note both Landsat 7 and 8 are denoted as band 2, 3, 4, 5, 6 and 7 ! ****
## i.e., follow the Landsat 8 band name convention 
import os
import datetime
import numpy as np

# import statistics 
# import json
import rasterio
from rasterio.windows import Window
import copy 

# (base) [hankui.zhang@cantrell hls_download]$ ls /weld/gsce_weld_9/HLS_v2/L30/2020/13/T/E/F/HLS.L30.T13TEF.2020033T173658.v2.0/
# HLS.L30.T13TEF.2020033T173658.v2.0.B01.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B02.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B03.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B04.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B05.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B06.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B07.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B09.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B10.tif  
# HLS.L30.T13TEF.2020033T173658.v2.0.B11.tif
# HLS.L30.T13TEF.2020033T173658.v2.0.cmr.xml
# HLS.L30.T13TEF.2020033T173658.v2.0.Fmask.tif
# HLS.L30.T13TEF.2020033T173658.v2.0.jpg
# HLS.L30.T13TEF.2020033T173658.v2.0.SAA.tif
# HLS.L30.T13TEF.2020033T173658.v2.0_stac.json
# HLS.L30.T13TEF.2020033T173658.v2.0.SZA.tif
# HLS.L30.T13TEF.2020033T173658.v2.0.VAA.tif
# HLS.L30.T13TEF.2020033T173658.v2.0.VZA.tif

# (base) [hankui.zhang@cantrell hls_download]$ ls /weld/gsce_weld_9/HLS_v2/S30/2020/13/T/E/F/HLS.S30.T13TEF.2020002T175729.v2.0/
# HLS.S30.T13TEF.2020002T175729.v2.0.B01.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B02.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B03.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B04.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B05.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B06.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B07.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B08.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B09.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B10.tif  
# HLS.S30.T13TEF.2020002T175729.v2.0.B11.tif
# HLS.S30.T13TEF.2020002T175729.v2.0.B12.tif
# HLS.S30.T13TEF.2020002T175729.v2.0.B8A.tif
# HLS.S30.T13TEF.2020002T175729.v2.0.cmr.xml
# HLS.S30.T13TEF.2020002T175729.v2.0.Fmask.tif
# HLS.S30.T13TEF.2020002T175729.v2.0.jpg
# HLS.S30.T13TEF.2020002T175729.v2.0.SAA.tif
# HLS.S30.T13TEF.2020002T175729.v2.0_stac.json
# HLS.S30.T13TEF.2020002T175729.v2.0.SZA.tif
# HLS.S30.T13TEF.2020002T175729.v2.0.VAA.tif
# HLS.S30.T13TEF.2020002T175729.v2.0.VZA.tif


class HLS_tile:
    """sentinel scene"""
    ## file names 
    # meta_file = ""
    input_files = ""
    qa_file = ""
    sat_file = ""
    sz_file = ""
    sa_file = ""
    vz_file = ""
    va_file = ""
    
    base_name = ""
    
    band_n = 6
    is_L8 = 0
    
    SIZE_30m = 3660
    # SIZE_10m = 10980
    
    ANGLE_SCALE = 100.0
    profile = 0
    tile_ID = 0
    year = 2019
    doy = 1
    
    # is_srf_nbar = 0 # 0 is srf and 1 is nbar 
    
    ## images 
    is_angle = False
    is_shadow = True
    is_toa = False 
    sz = 0
    sa = 0
    vz = 0
    va = 0
    # Landsat ARD
    # MULTI_SCALE = 0.0000275
    # ADD_SCALE = -0.2
    # HLS 
    MULTI_SCALE = 0.0001
    ADD_SCALE = 0
    
    THERMAL_SCALE = 0.01 
    
    reflectance_30m = np.empty([1])
    # reflectance_10m = np.empty([1])
    
    ## scl & is_valid are defined at 20 m 
    qa = np.empty([1])
    sat = np.empty([1])
    is_valid = np.empty([1])
    is_fill = np.empty([1])
    is_cloud = np.empty([1])
    is_adjacent = np.empty([1])
    is_shadow = np.empty([1])
    is_saturate = np.empty([1])
    is_snow = np.empty([1])
    is_water = np.empty([1])
    is_cirrus = np.empty([1])
    
    # def __init__(self,input_QA_file,is_L8=0):
    def __init__(self,input_QA_file, is_L8=1, is_angle=True, is_shadow=True, is_toa=False):
        # self.meta_file = ""
        self.is_L8 = is_L8
        self.qa_file = copy.deepcopy(input_QA_file)
        # self.sat_file = input_QA_file.copy().replace("QA_PIXEL.TIF","QA_RADSAT.TIF")
        self.sz_file  = copy.deepcopy(input_QA_file).replace("Fmask.tif","SZA.tif") # string.replace(oldvalue, newvalue, count) Yu Shen 09102024
        self.sa_file  = copy.deepcopy(input_QA_file).replace("Fmask.tif","SAA.tif")
        self.vz_file  = copy.deepcopy(input_QA_file).replace("Fmask.tif","VZA.tif")
        self.va_file  = copy.deepcopy(input_QA_file).replace("Fmask.tif","VAA.tif")
        
        self.is_angle = is_angle
        self.is_toa   = is_toa
        keywords = "SR"
        if is_toa:
            keywords = "TOA"
            MULTI_SCALE = 0.0000275
            ADD_SCALE = -0.2
        ## OH my god, they used the same scale for both TOA and surface 
        
        # self.input_files = np.full([band_n], fill_value=" ")
        self.input_files = list()
        if is_L8==1:
            # for i in range(band_n):
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B01.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B02.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B03.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B04.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B05.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B06.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B07.tif"))
          # self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B09.tif")) # Cirrus band in L30 of HLS Yu Shen 09102024
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B10.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B11.tif"))
            self.band_n = 9
        else:
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B01.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B02.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B03.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B04.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B8A.tif")) 
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B11.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B12.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B05.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B06.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B07.tif"))
            self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B08.tif"))
          # self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B09.tif")) # Water vapor band in S30 of HLS  Yu Shen 09102024 
          # self.input_files.append(copy.deepcopy(input_QA_file).replace("Fmask.tif","B10.tif")) # Cirrus in S30 of HLS  Yu Shen 09102024               
            self.band_n = 11
        
        self.base_name = os.path.basename(self.qa_file)
        self.tile_ID = self.base_name[8:14] 
        self.year = int(self.base_name[15:19])
        self.doy = int(self.base_name[19:22])
        # self.get_doy()
        
    
    # def get_doy(self):
        # year = int(self.base_name[15:19])
        # month = int(self.base_name[19:21])
        # day = int(self.base_name[21:23])
        # self.doy = (datetime.datetime(year=year, month=month, day=day)-datetime.datetime(year=year, month=1, day=1)).days+1
    
    def normalize_TOA(self):
        self.reflectance_30m[:,self.is_fill==0] = self.reflectance_30m[:,self.is_fill==0]/np.cos(self.sz[self.is_fill==0]/180*np.pi)
            # self.profile = src.profile
    
    def load_data(self,startx=0,starty=0,width=0,height=0):
        # self.meta_file = ""
        sz = 0
        sa = 0
        vz = 0
        va = 0
        
        ## if set up to 0, then it will read all the data domain.
        ## if not, then, it will read the spatial domain constrained by the window.
        if width==0 and height==0: 
            window=Window(0, 0, self.SIZE_30m, self.SIZE_30m)
        else: 
            # window=Window(startx, starty, width, height)
            # window=Window(row_off=starty, col_off=startx, height=height, width=width) # this is not like python array that height is dim[1] and width is dim[0]
            window=Window(row_off=startx, col_off=starty, height=width, width=height) # this is not like python array that height is dim[1] and width is dim[0]
  
        ## 30 m bands 
        # https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-2-science-products
        # self.reflectance_30m = np.full([self.band_n, self.SIZE_30m, self.SIZE_30m], fill_value=-9999, dtype=np.float32)
        self.reflectance_30m = np.full([self.band_n, window.height, window.width], fill_value=-9999, dtype=np.float32)

        ## initialize an array that is [n.3660,3660] ### Yu Shen 09102024
        for i in range(self.band_n):
            if not os.path.exists(self.input_files[i]):
                print ("file not exists "+self.input_files[i])
                return 0
            if self.is_L8==1 and i>=7:
                with rasterio.open(self.input_files[i]) as src:
                    ### adjusting two thermal bands in L30 in HLS ### Yu Shen 09102024
                    self.reflectance_30m[i,:,:] = src.read(window=window).astype(np.float32)*self.THERMAL_SCALE 
                    self.profile = src.profile
            else:
                with rasterio.open(self.input_files[i]) as src:
                    ### adjusting all non-thermal bands in L30 or S30 in HLS ### Yu Shen 09102024
                    self.reflectance_30m[i,:,:] = src.read(window=window).astype(np.float32)*self.MULTI_SCALE+self.ADD_SCALE
                    self.profile = src.profile
        
        
        # if not os.path.exists(self.sat_file):
            # print ("file not exists "+self.sat_file)
            # return 0
        
        # with rasterio.open(self.sat_file) as src:
            # self.sat = src.read()[0,:,:]
        
        if self.is_angle:
            ## sz
            if not os.path.exists(self.sz_file):
                print ("file not exists "+self.sz_file)
                return 0
            ### adjusting all angle related bands in HLS into decimal range ### Yu Shen 09102024
            with rasterio.open(self.sz_file) as src:
                self.sz = src.read(window=window)[0,:,:].astype(np.float32)/self.ANGLE_SCALE
            
            ## sa
            if not os.path.exists(self.sa_file):
                print ("file not exists "+self.sa_file)
                return 0
            
            with rasterio.open(self.sa_file) as src:
                self.sa = src.read(window=window)[0,:,:].astype(np.float32)/self.ANGLE_SCALE
            
            ## vz
            if not os.path.exists(self.vz_file):
                print ("file not exists "+self.vz_file)
                return 0
            
            with rasterio.open(self.vz_file) as src:
                self.vz = src.read(window=window)[0,:,:].astype(np.float32)/self.ANGLE_SCALE
            
            ## va
            if not os.path.exists(self.va_file):
                print ("file not exists "+self.va_file)
                return 0
            
            with rasterio.open(self.va_file) as src:
                self.va = src.read(window=window)[0,:,:].astype(np.float32)/self.ANGLE_SCALE
        
        ## cloud mask ARD 
        ## not work anymore below 
        ## https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1337_Landsat7ETM-C2-L2-DFCB-v5.pdf
        ## https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf
        ## below works on Jul 7 2022
        ## -> https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1337_Landsat7ETM-C2-L2-DFCB-v5.pdf
        ## -> https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf
        
        ## cloud mask HLS  
        if not os.path.exists(self.qa_file):
            print ("file not exists "+self.qa_file)
            return 0
        with rasterio.open(self.qa_file) as src:
            self.qa = src.read(window=window)[0,:,:]
            self.is_fill    = self.reflectance_30m[0,:,:]==(-9999.*self.MULTI_SCALE+self.ADD_SCALE) # 1 is filled and 0 is image
            self.is_cloud   = np.bitwise_and(np.right_shift(self.qa,1),1) # bit 1 cloud 
            self.is_adjacent= np.bitwise_and(np.right_shift(self.qa,2),1) # bit 2 Adjacent to cloud/shadow 
            self.is_shadow  = np.bitwise_and(np.right_shift(self.qa,3),1) # bit 3 cloud shadow
            self.is_snow    = np.bitwise_and(np.right_shift(self.qa,4),1) # bit 4 snow/ice
            self.is_water   = np.bitwise_and(np.right_shift(self.qa,5),1) # bit 5 water
            # np.logical_or.reduce((np.bitwise_and(np.right_shift(self.qa,1),1),np.bitwise_and(np.right_shift(self.qa,3),1) )) # cloud 
        
        is_not_valid = np.logical_or.reduce((self.is_fill,self.is_cloud,self.is_adjacent,self.is_shadow,self.is_snow))
        # self.reflectance_30m[:,self.is_fill] = -9999.
        ### make all the cloud/adjacent-cloud/shadow/snow pixels to be -9999 I guess ### Yu Shen 09102024
        self.reflectance_30m[:,is_not_valid] = -9999. 
        self.is_valid = np.logical_not(is_not_valid)
        return 1
        # scl = np.full([self.SIZE, self.SIZE], fill_value=0, dtype=np.int16)
        # is_valid = np.full([self.SIZE, self.SIZE], fill_value=0, dtype=np.int16)
    
    ## for test only
    def save_valid_file (self):
        naip_meta = self.profile
        naip_meta['dtype'] = 'uint8'
        naip_meta['count'] = 1
        with rasterio.open("./valid."+self.base_name, 'w', **naip_meta) as dst:
            dst.write(self.is_valid.astype(np.uint8), indexes=1 )    
    
    ## for test only
    # image = valid_sum
    def save_image_file (self,image,prefix="total_n",folder="./"):
        with rasterio.open(self.qa_file) as src:
            profile = src.profile        
        
        naip_meta = profile
        naip_meta['tiled'] = False
        naip_meta['compress'] = 'LZW'
        if image.dtype==np.uint8 or image.dtype==np.int8 :
            naip_meta['dtype'] = 'uint8'
        elif image.dtype==np.float32:
            naip_meta['dtype'] = 'float32'
        
        ## replace base_name with tile_ID. 11/08/2024
        ## add an extention ".tif" to this filename
        file_name = folder+prefix+self.tile_ID + '.tif'
        if len(image.shape)==2:
            naip_meta['count'] = 1
            with rasterio.open(file_name, 'w', **naip_meta) as dst:
                dst.write(image, indexes=1 )    
        else: 
            naip_meta['count'] = image.shape[0]
            with rasterio.open(file_name, 'w', **naip_meta) as dst:
                dst.write(image)    
    
  
    
