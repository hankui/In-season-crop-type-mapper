# -*- coding: utf-8 -*-
"""
@author: Hankui Zhang and Yu Shen 

Near real-time crop type mapper created based on Hank (hankui.zhang@sdstate.edu) CRIT model.

"Classifying raw irregular time series (CRIT) for large area land cover mapping by adapting transformer model" 
https://doi.org/10.1016/j.srs.2024.100123
"""

"""
================== version history log =======================================================================
# v5_1 Using input parameters so that this is ready for release------------------------------------Jan 21 2025
# v5_0 Using only one year of historical HLS obs (i.e., two years of HLS) as input. ---------------Oct 15 2024
# v4_0 Improve the code efficiency by adjusting BATCH size, Chunck size, input data size, ... ---- Oct 11 2024 
       The time decreased from 28 hrs to 14 hrs when temporal and spatial gaps are removed. ------------------
       The time decreased from 14 hrs to 8  hrs when CHUNCK size increased from 610 to 915 and BATCH size ----
       increased from 1024 to 2048. Not sure if change np.float32 to np.float16 will improve the efficiency. - 
# v4_0 Hank checked Yu's codes ------------------------------------------------------------------- Oct 07 2024
# v3_8 Yu Shen has modified this code for facilliate the HLS inputs. Some errors may cause. ------ Sep 19 2024
# v3_7 final version - note there is a potential issue of legend of class 8 (Barren), ------------------------
       NO issue as the legend dsr (lcmap.dsr) has been adjusted to this --------------------------------------
# v3_5 use new model with 35 years + full training sample model ----------------------------------------------
# v3_4 use new model with 35 years ---------------------------------------------------------------------------
# v3_2 use subset innovator + multiple years ----------------------------------------------------- Jan 21 2024
# v3_1 use subset innovator ---------------------------------------------------------------------- Jan 21 2024 
# v3_0 use subset innovator ---------------------------------------------------------------------- Jan 21 2024 
# v2_5 use new server innovator ------------------------------------------------------------------ Jan 21 2024 
# v2_4 plot using the small n legend (min1max7step1is_minTRUEis_maxFALSE.dsr) --------------------------------
# v2_3 same as v2_2 but change the true color composite into more months -------------------------------------
# v2_2 use a long time series model rather than keeping splitting for solution with no. of observations >80 --
# v2_1 fix some to keep split the tiles until the tiles reach 80 ---------------------------------------------
# v2_0, to use DEM  ------------------------------------------------------------------------------ Apr 20 2023
# v1_5, to run different batches ----------------------------------------------------------------- Jan 04 2023
# v1_2/3/4, to run a model with daily input ------------------------------------------------------ Jan 04 2023
# v1_1, v1_1 to fix batch size, add number of clear browses -------------------------------------- Dec 05 2022
# v1_0, v1_0 to run the model Pro_load_model_run_tile_v1_0.py ------------------------------------ Dec 05 2022
# ------------------ Notes -----------------------------------------------------------------------------------
# in ARD coordinates x<0, y>0; a pixel in h18v18 (-152,264,  576,174) ----------------------------------------
# import Pro_load_model_run_tile_v3_0 ------------------------------------------------------------------------
================== version history log ======================================================================= 
""" 

## standard python library
import sys
import os 
import numpy as np 
import importlib      
import gc 
import tensorflow as tf
from datetime import datetime
import calendar

## self-developed library
import find_files
import transformer_encoder44
import config
import HLS_io_chunks
import color_display # for storing the output file
importlib.reload(find_files)
importlib.reload(HLS_io_chunks)

np.set_printoptions(suppress=True) # turn off scientific notation  

## ============================================================================================================
## ******************************************** Default parameters ********************************************
## ============================================================================================================

## ==== fixed parameters =================
FILL = -9999.0    # fill value
BANDS_N = 12      # @YuShen 09202024 11 Sentinel-2 SpectralBands + 1 (DOY) bands
XY_DIM = 3660     # The HLS tile dimenssion (3660 X 3660) ~ 110 km * 110 km  ## @YuShen 09192024
PERIODS = 176     # e.g.: 14SPG in 2021 has files of 181; 14TLT has 203 files in 2021/2022/2023               
# LONG_PERIODS = 210 ## what does this mean? max-length 176? 09192024
tile_id = ""      # format: "14TLT"
is_true_color = False
N_YEAR_WINDOW = 2 # The input here is 2-yrs HLS historical Obs + Obs in current year

## ==== adjustable parameters ============
BATCH = 2048      # It seems like the batch size is 512 in last talk, adjustable in the future
XY_DIM_SUB = 1220 # Make sure this number is divisible by XY_DIM = 3660; otherwise, the dimension of process_data_i has to be changed dynamically
                  # == 1220 exceed %10 memory. 
FIRST_YEAR = 2022 # specify the start year and the prediction year?? No 
Month = -1        # to specify the latest month in current year to simulate the real-time prediction. 
                  # if Month ==  6, then the NRT crop map is predicted using the HLS observations before the last day in June (6).
DOYset =-1        # if Month is given, then DOYset is set to -1. If DOYset is assigned to a number from 1-365/366, then the "Month" will not be used.                  

INPUTDIR_ROOT = "HLS_RAW_DATA_DIRECTORY"  # input HLS raw data directory
CLASS_DIR  = "/HLS_NRT_CLASSIFICATION_OUTPUT/" # output for HLS Land cover predictions
BROWSE_DIR = "/A_SUB_DIRECTORY_FOR_CHECKING/" # output for thematic HLS Land cover predictions with color-legend assigned
  
## ==== other parameters =================
from config import MAX_LANDSAT,MAX_SENTINEL2,L8_fields,S2_fields,mapping
L8_bands_n = len(L8_fields)//MAX_LANDSAT   
S2_bands_n = len(S2_fields)//MAX_SENTINEL2
   
version='v5_0'
if '__file__' in globals():
    print(os.path.basename(__file__))
    version = os.path.basename(__file__)[24:28] 
    print(version) 

## ============================== load trained model ==========================================================
mean_name  = "./v1_70.layer4.METHOD2.BATCH64.LR0.0002.EPOCH30.L20.1_mean.csv"
model_path = "./v1_70.layer4.METHOD2.BATCH64.LR0.0002.EPOCH30.L20.1.i0.model.h5"

x_mean = 0
x_std  = 1
if os.path.exists(mean_name):
    dat_temp = np.loadtxt(open(mean_name, "rb"), dtype='<U30', delimiter=",", skiprows=1)
    arr = dat_temp.astype(np.float64)
    x_mean,x_std = arr[:,0],arr[:,1]
else:
    print("Error !!!! mean file not exists " + mean_name)

## ============================================================================================================
## *************************************************** FUNCTIONS **********************************************
## ============================================================================================================

## ===== for a given HLS filelist, it returns unique day of year list ("2020030","2020044",...) and its length
# filenames = yrs_flst; prefix = tile_id
def count_unique_DOYs(filenames, prefix):
    unique_days = set()  # A set to store unique "day of year" parts
    for file in filenames:
        ## HLS.L30.T13TEF.2020033T173658.v2.0.Fmask.tif
        # Check if the file contains the specified prefix
        if prefix in file:
            # Find the position where the 6-character prefix appears
            file_base = os.path.basename(file)
            start_index = file_base.index(prefix) + len(prefix)  # Get the start index after the prefix
            if start_index + 7 <= len(file_base):  # Ensure there's enough length to extract the YYYYDDD part
                day_of_year = file_base[start_index+1:start_index + 8]  # Extract the 7 characters YYYYDDD
                unique_days.add(day_of_year)  # Add the day_of_year to the set
    
    # Return both the set of unique DOYs and the count of unique days
    unique_days_list = list(unique_days)
    return unique_days_list, len(unique_days)

## separate filenames to L30 list and S30 list and make their DOYs aligned to each other if both of them have files in same day
# yr_L30_flst,yr_S30_flst,yr_L30_DOY,yr_S30_DOY = separate_filenames(yrs_flst, total_length = PERIODS, tile_id = tile_id)
def separate_filenames(filenames, total_length=PERIODS, tile_id = "14TLT"):
    # Initialize two lists with "FILL" for both L30 and S30
    l30_files = ["FILL"] * total_length
    s30_files = ["FILL"] * total_length
    l30_doy = ["FILL"] * total_length
    s30_doy = ["FILL"] * total_length
    # Dictionary to store day of year (DOY) and their respective indices
    doy_to_index = {}
    index_counter = 0
    # Iterate through filenames to extract DOY and organize files
    for file in filenames:
        file_base = os.path.basename(file)
        # Extract the YYYYDDD (day of year) part ## ## HLS.L30.T13TEF.2020033T173658.v2.0.Fmask.tif
        doy_part = None
        for i in range(len(file_base) - 6):
            if file_base[i:i+7].isdigit():
                doy_part = file_base[i+4:i+7] ## if only return 2021013, use file[i:i+7]
                break
        
        # If a new DOY is found, map it to the next available index
        if doy_part not in doy_to_index and index_counter < total_length:
            doy_to_index[doy_part] = index_counter
            index_counter += 1
        
        # Now place the file in the correct list based on "L30" or "S30"
        if "L30" in file and doy_part in doy_to_index:
            l30_files[doy_to_index[doy_part]] = file
            l30_doy[doy_to_index[doy_part]] = doy_part
        elif "S30" in file and doy_part in doy_to_index:
            s30_files[doy_to_index[doy_part]] = file
            s30_doy[doy_to_index[doy_part]] = doy_part
    
    # Return both lists (L30 and S30) and their corresponding DOY lists
    return l30_files, s30_files, l30_doy, s30_doy

# selection of some typical HLS tiles for intercomparison or illustration purposes
def golden_tiles(tileid):
    return tileid=="14TLT" or tileid=="14TNQ" or tileid=="14TPL" or tileid=="14SNH" or tileid=="15TVL" or tileid=="15TWH" or tileid=="16TCK" or tileid=="16TCP"  or tileid=="16TFN"

# To get the day of the year for the last day of a given month and year
def get_end_day_of_year(year, month):
    import datetime
    # Get the last day of the given month and year
    last_day = calendar.monthrange(year, month)[1]  # Get the number of days in the month
    # Create a date object for the last day of the month
    end_date = datetime.date(year, month, last_day)
    # Get the day of the year
    day_of_year = end_date.timetuple().tm_yday
    return day_of_year

def get_filelist_before_given_date (flist, threshold):
    flistout = []
    for file in flist:
          file_base = os.path.basename(file)
          # Check if the filename is not "FILL"
          if file != "FILL":
              # Extract the date part from the filename
              date_str = file_base.split(".")[3]   # Example: '2023164T175639'
              date_value = int(date_str[:7])  # Get the first 7 characters as an integer (e.g., 2023164)
              # Check if the date is before the threshold
              if date_value < threshold:
                  flistout.append(file)
              else:
                  flistout.append("FILL")
          else:
              flistout.append("FILL")
    return flistout

## model_path; periods= COMMON_DAYS_BLOCK_N
def load_model(model_path, periods=PERIODS):
    # Keras is an open-source deep learning library written in Python.
    # tf.keras API brings Keras’s simplicity and ease of use to the TensorFlow project.
    importlib.reload(transformer_encoder44)
    model = tf.keras.models.load_model(model_path, compile=False) 
    if periods==PERIODS:
        return model
    
    importlib.reload(config)
    from config import MAX_LANDSAT,MAX_SENTINEL2,YEARS_DATA
    L8_bands_n = len(L8_fields)//MAX_LANDSAT   
    S2_bands_n = len(S2_fields)//MAX_SENTINEL2 
    
    layern1 = 3
    layern2 = 4
    units = 256
    head = 8
    drop = 0.1
    is_day_input = 1
    concat = 4
    n_class = 50 
    model_long = transformer_encoder44.get_transformer_reflectance_cdl(MAX_LANDSAT=periods,MAX_SENTINEL2=periods,L8_bands_n = L8_bands_n, S2_bands_n = S2_bands_n, \
                                                                       YEARS_DATA = YEARS_DATA,layern1=layern1, layern2=layern2, units=units, \
                                                                       is_reflectance=False, n_head=head, drop=drop, \
                                                                       is_day_input=is_day_input, is_sensor=True, is_xy=False,\
                                                                       concat=concat, n_out=n_class)   
    embedding_name = ""
    for il,ilayer in enumerate(model.layers):
        ilayer1 = model     .layers[il] 
        ilayer2 = model_long.layers[il] 
        # if (model_drop==0 and 'dropout' not in ilayer2.name) or model_drop>0: # to handle one model has dropout while the other does no
            # il1=il1+1 
        # else:
            # continue 
        # ilayer1 = model    .layers[il1] 
        name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic!='_']) 
        name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic!='_']) 
        if "embedding" in name_cls:
            embedding_name = ilayer1.name
        if name_cls==name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
            # print ("\t"+ilayer.name, end=" ")
            model_long.layers[il].set_weights (model.layers[il].get_weights())
    return model_long

## ============================================================================================================ 
## *************************************************** MAIN PRO ***********************************************
## ============================================================================================================ 
## run time example:
# INPUTDIR_ROOT = "/mmfs1/scratch/jacks.local/yu.shen2069/Data/HLS/Raw"  # input HLS raw data directory
# CLASS_DIR  = "/mmfs1/scratch/jacks.local/hankui.zhang/workspace/hls_transformer/test_output/" # output for HLS Land cover predictions
# tile_ID         =   "T14TLT"
# FIRST_YEAR      =   2022
# DOYset          =   190
## python Pro_load_model_run_tile_v5_1.py /mmfs1/scratch/jacks.local/yu.shen2069/Data/HLS/Raw/ /mmfs1/scratch/jacks.local/hankui.zhang/workspace/hls_transformer/test_output/ T14TLT 2023 190

if __name__ == "__main__":
    
    ### input tiles, satellites, and years:
    # 222 tiles to cover above 9 states
    ## ===================== North Dakota =========================== ## done
    # "T13UEQ","T13UFQ","T13UGQ","T14ULV","T14UMV","T14UNV","T14UPV",
    # "T13UEP","T13UFP","T13UGP","T14ULU","T14UMU","T14UNU","T14UPU",
    # "T13TEN","T13TFN","T13TGN","T14TLT","T14TMT","T14TNT","T14TPT",
    # "T13TEM","T13TFM","T13TGM","T14TLS","T14TMS","T14TNS","T14TPS"
    
    ## ===================== South Dakota =========================== ## done
    # "T13TEL","T13TFL","T13TGL","T14TLR","T14TMR","T14TNR","T14TPR",
    # "T13TEK","T13TFK","T13TGK","T14TLQ","T14TMQ","T14TNQ","T14TPQ",
    # "T13TEJ","T13TFJ","T13TGJ","T14TLP","T14TMP","T14TNP","T14TPP",
    # "T13TEH","T13TFH","T13TGH","T14TLN","T14TMN","T14TNN","T14TPN"
    
    ## ===================== Nebraska =============================== ## done
    # "T13TEG","T13TFG","T13TGG","T14TKM","T14TLM","T14TMM","T14TNM","T14TPM","T14TQM",
    # "T13TEF","T13TFF","T13TGF","T14TKL","T14TLL","T14TML","T14TNL","T14TPL","T14TQL",
    # "T13TGE","T14TKK","T14TLK","T14TMK","T14TNK","T14TPK","T14TQK"
    
    ## ===================== Kasas ================================== ## done
    # "T13SGD","T14SKJ","T14SLJ","T14SMJ","T14SNJ","T14SPJ","T14SQJ","T15STD","T15SUD",
    # "T13SGC","T14SKH","T14SLH","T14SMH","T14SNH","T14SPH","T14SQH","T15STC","T15SUC",
    # "T13SGB","T14SKG","T14SLG","T14SMG","T14SNG","T14SPG","T14SQG","T15STB","T15SUB"
    
    ## ===================== Oklahoma =============================== ## done
    # "T13SFA","T13SGA","T14SKF","T14SLF","T14SMF","T14SNF","T14SPF","T14SQF","T15STA","T15SUA",
    # "T14SME","T14SNE","T14SPE","T14SQE","T15STV","T15SUV",
    # "T14SMD","T14SND","T14SPD","T14SQD","T15STU","T15SUU",
    # "T14SMC","T14SNC","T14SPC","T14SQC","T15STT","T15SUT"
    
    ## ===================== Minnesota =============================== ## done
    # "T14UQV","T15UUQ","T15UVQ","T14UQU","T15UUP","T15UVP","T15UWP","T15UXP","T15UYP",
    # "T14TQT","T15TUN","T15TVN","T15TUN","T15TWN","T15TXN",
    # "T14TQS","T15TUM","T15TVM","T15TWM","T14TQR","T15TUL","T15TVL","T15TWL",
    # "T14TQQ","T15TUK","T15TVK","T15TWK","T14TQP","T15TUJ","T15TVJ","T15TWJ","T15TXJ"
    
    ## ===================== Iowa ==================================== ## under 
    # "T14TQN","T15TUH","T15TVH","T15TWH","T15TXH",
    # "T15TTG","T15TUG","T15TVG","T15TWG","T15TXG","T15TYG",
    # "T15TTF","T15TUF","T15TVF","T15TWF","T15TXF","T15TYF"
    
    ## ===================== Missouri ================================ ## under
    # "T15TTE","T15TUE","T15TVE","T15TWE","T15TXE",
    # "T15SVD","T15SWD","T15SXD","T15SYD",
    # "T15SVC","T15SWC","T15SXC","T15SYC",
    # "T15SVB","T15SWB","T15SXB","T15SYB","T16SBG",
    # "T15SVA","T15SWA","T15SXA","T15SYA","T16SBF"
    
    ## ===================== Arkansas =============================== ## under
    # "T15SVV","T15SWV","T15SXV","T15SYV","T16SBE",
    # "T15SVU","T15SWU","T15SXU","T15SYU",
    # "T15SVT","T15SWT","T15SXT",
    # "T15SVS","T15SWS","T15SXS"
    
    ## ============ Oklahoma ===============================
    tiles_array = ["T13SFA","T13SGA","T14SKF","T14SLF","T14SMF","T14SNF","T14SPF","T14SQF","T15STA","T15SUA","T14SME","T14SNE","T14SPE","T14SQE","T15STV","T15SUV","T14SMD","T14SND","T14SPD","T14SQD","T15STU","T15SUU","T14SMC","T14SNC","T14SPC","T14SQC","T15STT","T15SUT"]
    
    
    # starts from May-1 with each 15-day interval, end at Aug-7
    # for tiles of ["T14UNU", "T14TNT", "T14TPS", "T14TPR", "T14TQQ", "T15TUJ", "T15TVH", "T15TWG", "T15TXF"]
    # ["T16TBK","T16SCJ","T16SCH","T16SDG","T16SCF","T16SBE","T15SXU","T15SYT","T15SXS","T15SXR"]
    
    ## input parameters 
    INPUTDIR_ROOT   =      (sys.argv[1])
    CLASS_DIR       =      (sys.argv[2])
    tile_ID         =      (sys.argv[3])
    FIRST_YEAR      =   int(sys.argv[4])-1
    DOYset          =   int(sys.argv[5])
    tiles_array = list()
    tiles_array.append(tile_ID) 
    # IF Month is assigned (to be non -1), please set up the DOYset to be -1. IF DOYset is assigned (to be non -1), please set up the Month to be -1
    Month = -1 
    # CLASS_DIR = CLASS_DIR+"/"
    if not os.path.isdir(CLASS_DIR): 
        os.makedirs(CLASS_DIR)      
    
    BROWSE_DIR = CLASS_DIR+"/browse/"
    if not os.path.isdir(BROWSE_DIR): 
        os.makedirs(BROWSE_DIR)
    
    if DOYset<=0 or DOYset>366:
        print ("DOYset is out of range !")
        exit() 
    
    # DOYset= -1
    
    # L2              = float(sys.argv[5])
    if len(sys.argv)>6:
        BATCH  =   int(sys.argv[6])    
    
    print ("BATCH_SIZE =\t" + str(BATCH))
    

    # DOYsets = [65,51,37] 
    # Months = [7] # 6,7,8,9,10,11
    # for Month in Months: 
    # for DOYset in DOYsets:
    for tile_name in tiles_array:
        
        # tile_name = tiles_array[0]
        tile_id = tile_name[1:]
        
        print ("\nThe current HLS tile is :", tile_id, "...\n")
        print ("\nThe current DOY set  is :", DOYset , "...\n")
        
        # class_file_list = find_files.find_list(CLASS_DIR+"/", is_L8=False, pattern=tile_id+"_"+str(FIRST_YEAR) )
        # if len(class_file_list)>0: 
        #    print ("! The class file has been generated " + tile_id)
        #    print (class_file_list)
        #    # exit(1) # this need to be changed later 
        
        start = datetime.now() ## recording the time spent
        print_str = '\n\n\nstart time: '+ str(start) # + '\nstation sentinel1_folder #_img vgt_cdl' +'\n======================================'
        
        class_image = np.full([XY_DIM,XY_DIM],fill_value=255,dtype=np.int8) 
        
        ## ====================================================================================================================== #
        ## ====================================== Arrange L30/S30 filename ====================================================== #
        ## ==== get three years of L30 and S30 and their DOYs with the format of L8yr1, S2yr1, L8yr2, S2yr2, L8yr3, S2yr3======== #
        ## ====================================================================================================================== #
        ## Step1: Find the largest number of HLS observations in three years, then assign PERIODS to this #
        
        for yr in range(FIRST_YEAR,FIRST_YEAR+N_YEAR_WINDOW):
            Filelist_L30 = find_files.find_L30_qa_list_years(INPUTDIR_ROOT+"/T"+tile_id+'/', is_L8=False, \
                                                             sensor = "L30",pattern=".Fmask.tif",tile_id=tile_id,\
                                                             starty=yr,endy=yr+1)
            Filelist_S30 = find_files.find_S30_qa_list_years(INPUTDIR_ROOT+"/T"+tile_id+'/', is_L8=False, \
                                                             sensor = "S30",pattern=".Fmask.tif",tile_id=tile_id,\
                                                             starty=yr,endy=yr+1)
            yrs_flst =  Filelist_L30 + Filelist_S30
            DOY_list,total_n_yr = count_unique_DOYs(yrs_flst, tile_id)
            if total_n_yr >= PERIODS:
                PERIODS = total_n_yr
                
        ## Step2: Assume each year has PERIODS number of Observations, assign L30 and S30 as: 
        ## === yr1_L30 + yr1_S30 + yr2_L30 + yr2_S30 + yr3_L30 + yr3_S30 + ... 'FILL' =======
        ALL_2yrs_filename = []
        for yr in range(FIRST_YEAR,FIRST_YEAR+N_YEAR_WINDOW):
            Filelist_L30 = find_files.find_L30_qa_list_years(INPUTDIR_ROOT+"/T"+tile_id+'/', is_L8=False, \
                                                             sensor = "L30",pattern=".Fmask.tif",tile_id=tile_id,\
                                                             starty=yr,endy=yr+1)
            Filelist_S30 = find_files.find_S30_qa_list_years(INPUTDIR_ROOT+"/T"+tile_id+'/', is_L8=False, \
                                                             sensor = "S30",pattern=".Fmask.tif",tile_id=tile_id,\
                                                             starty=yr,endy=yr+1)
            yrs_flst =  Filelist_L30 + Filelist_S30
            # DOY_list,total_n_yr = count_unique_DOYs(yrs_flst, tile_id)
            # total_N2 += total_n_yr  
            tempflist = (yrs_flst).copy(); tempflist.sort()
            yr_L30_flst,yr_S30_flst,yr_L30_DOY,yr_S30_DOY = separate_filenames(tempflist, total_length = PERIODS, tile_id = tile_id)
            ALL_2yrs_filename += (yr_L30_flst + yr_S30_flst) ## three years of L30+S30 filename. (includes FILL filenames)
            
        
        ## For 14TLT, there are all 1218 filenames strings (include "FILL"), 1218 = 203 * 2 * 3
        ## as the max_len(yearly_filenamelists) is 203 for a year in (2021,2022,2023)
        total_n = len(ALL_2yrs_filename)
        
        ## set up the current date as "threshold_date", assign all the files after this date to be "FILL" to simulate the
        ## real time situation. (1) by giving the month; (2) by giving the day-of-year
        if (Month != -1): 
            END_DOY_Givenmonth = get_end_day_of_year((FIRST_YEAR+N_YEAR_WINDOW-1), Month)
            print("\nThe day of the year for the first day of ", Month, "/", str(FIRST_YEAR+N_YEAR_WINDOW-1), "is :", END_DOY_Givenmonth, "\n")
        
        if (DOYset != -1):
            print("\nThe DOYset is ",DOYset, "\n")
        
        if DOYset == -1 and Month != -1:
            threshold_date = int(str(FIRST_YEAR+N_YEAR_WINDOW-1) + str(END_DOY_Givenmonth).zfill(3))
            print("\nthreshold date is DOY == ", threshold_date, "\n")
            ALL_2yrs_filename = get_filelist_before_given_date(ALL_2yrs_filename,threshold_date)
        
        if DOYset != -1 and Month == -1:
            threshold_date = int(str(FIRST_YEAR+N_YEAR_WINDOW-1) + str(DOYset).zfill(3))
            print("\nthreshold date is DOY == ", threshold_date, "\n")
            ALL_2yrs_filename = get_filelist_before_given_date(ALL_2yrs_filename,threshold_date)
        
        ## =========================================================================================================== 
        ## ========================== Get the big array of process_data_i using the HLS loaded data ================== 
        ## =========================================================================================================== 
        
        # [10/11/2024] try to use np.float16 rather than 32 to decrease loaded data size and increase chunck size at the same time.
                              ## and add muliprocessing and more workers for improving efficiency perhaps. 
                              ## It seems like this kind of adjustment does't work.
        # [10/10/2024] try to increase chunck size from 610 to 915 and BATCH from 1024 to 2048, time decreased 4 hours (12 to 8 hrs).
        
        doys  = np.full([total_n],fill_value=FILL,dtype=np.float32)
        years = np.full([total_n],fill_value=FILL,dtype=np.float32)
        valid_year_doys = np.full([total_n],fill_value=False,dtype=bool)
        valid_year_doys_union = np.full([total_n],fill_value=False,dtype=bool)
       
        # process_data_i = np.full([XY_DIM_SUB*XY_DIM_SUB,LONG_PERIODS*N_YEAR_WINDOW,BANDS_N+2+XY_DIM_N+1],fill_value=-9999.0,dtype=np.float32) @YuShen 09202024
        process_data_i = np.full([XY_DIM_SUB*XY_DIM_SUB,PERIODS*N_YEAR_WINDOW*2,BANDS_N],fill_value=FILL,dtype=np.float32) 
        
        # 57.0938 GB for 14TLT  in 2021 - 2023 (580 files) 
        # 114.1875 GB for all data size.
        # all_data size GB 95.8426 for 
        # input all raw data size GB 391.5802
        # print ("all raw data size GB: {:5.4f}".format(sys.getsizeof(process_data_i)/XY_DIM_SUB/XY_DIM_SUB)) 
        
        # Calculate the size in gigabits
        size_in_bytes = process_data_i.nbytes
        size_in_gigabits = (size_in_bytes * 8) / 1e9
        print ("input all raw data size GB {:5.4f}".format(size_in_gigabits))
        
        ## ============================================================================================================
        ## ====================================== chunks processing ===================================================
        ## ============================================================================================================
        
        # XY_DIM = XY_DIM_SUB # to test the time spent for each block
        
        # outi = 0; outj = 0; widthi=XY_DIM_SUB; heightj=XY_DIM_SUB
        for outi in range(0,XY_DIM,XY_DIM_SUB):
            widthi = min(XY_DIM_SUB,XY_DIM-outi)
            # print("outi = {:d} and widthi = {:d}".format(outi, widthi), end="\t")
            for outj in range(0,XY_DIM,XY_DIM_SUB):
                heightj = min(XY_DIM_SUB,XY_DIM-outj)
                # print("\toutj = {:d} and heightj = {:d}".format(outj, heightj), end="\t\n")
                
                all_data = np.full([widthi,heightj,total_n,BANDS_N],fill_value=FILL,dtype=np.float32)
                all_qa   = np.full([widthi,heightj,total_n,       ],fill_value=False  ,dtype=bool   )
                
                for i in range(total_n):
                    if ALL_2yrs_filename[i] != "FILL": 
                        ## Basically here to make all the HLS data into an array of [XY_DIM_SUB^2,total_n,12] if all 
                        ## pixels have valid values. HLSi may be the HLS L30 file and may also be HLS S30 file.
                        HLSi = HLS_io_chunks.HLS_tile(ALL_2yrs_filename[i],is_L8 = 'HLS.L30' in ALL_2yrs_filename[i])
                        HLSi.load_data(outi,outj,widthi,heightj)
                        fyear = int(HLSi.year)
                        # if HLSi.is_valid.sum()>0 and fyear==2023: # debug only 
                            # break;
                        
                        valid_year_doys[i] = HLSi.is_valid.sum()>0 
                        # adding the doy value at the first band            ## 01th BANDS 
                        if str(fyear).strip() in ALL_2yrs_filename[i]:
                            all_data[HLSi.is_valid,i,0] = int(fyear-FIRST_YEAR) + int(HLSi.doy-1) / 366.0 ## for DOY
                        else:
                            print ('!!! Not L30 str(fyear).strip() in ALL_2yrs_filename[i] ')
                        
                        all_qa[HLSi.is_valid,i,] = True  ## QA
                        if "L30" in ALL_2yrs_filename[i]: 
                            # all_data[HLSi.is_valid,i,L8_bands_n:] = FILL  ## 10-11th BANDS
                            for bi in range(1,(BANDS_N-4)): ##[1,BANDS_N-4] ## 01-08th BANDS
                                # all_data[landsati.is_valid,i,bi] = (landsati.reflectance_30m[bi,landsati.is_valid]-x_mean[bi])/x_std[bi] 
                                all_data[HLSi.is_valid,i,bi] = (HLSi.reflectance_30m[bi-1,HLSi.is_valid]-x_mean[bi])/x_std[bi]     ## 00-08
                        else:
                            for bi in range(1, BANDS_N):   ##[1,BANDS_N-1]  ## 01-11th BANDS 
                                # all_data[landsati.is_valid,i,bi] = (landsati.reflectance_30m[bi,landsati.is_valid]-x_mean[bi])/x_std[bi] 
                                all_data[HLSi.is_valid,i,bi] = (HLSi.reflectance_30m[bi-1,HLSi.is_valid]-x_mean[bi+8])/x_std[bi+8] ## 08-19
                                
                        years[i] = HLSi.year-FIRST_YEAR # 0, 1
                        doys [i] = HLSi.doy
                        # break 
                    # break 
                # break # In debug mode, each chunk [512,512] can be output for checking potential error 
                               
    ## ============================================================================================================
    ## ============================== Adjusting the design if the max HLS obs in a year exceeds 176 ===============
    ## ============================================================================================================
                ## no_of_obs_image[:,:] each pixel value is the sum of true values along time (N_years_window)
                no_of_obs_image = all_qa.sum(axis=(2)) # 2 is the dimension of the time 
                
                ## The number of valid pixels, where each pixel has at least one valid ref along time.
                ## The time series numbers. Each time series has at least one valid ref. along three years.
                # valid_pixel_image = np.logical_and.reduce(no_of_obs_image>0)    
                valid_pixel_image = no_of_obs_image>0
                valid_pixel_time  = all_qa.sum(axis=(0,1))>0
                # all_data[valid_pixel_image,:,1][:1,valid_pixel_time]
                    
                process_n = valid_pixel_image.sum()  # (valid time series length (at least one valid reflectance) )
                if process_n==0:
                    continue 
                # print ("\t\t no_of_obs_image.max()//N_YEAR_WINDOW = {:d}, MAX_TIME_SERIES_YEAR = {:d}, \
                # PERIODS_INPUT = {:d}, BATCH={:d}".format(no_of_obs_image.max()//N_YEAR_WINDOW,PERIODS,PERIODS,BATCH) ) 
                
                ## ====== Exclude those paired L30 and S30 files that have no valid observations based on QA ======
                ## ====== By reduce the input file length from PERIODS to max_n_year ==============================
                ## ====== Max_n_year is the max length of L30 and S30 pairs where =================================
                ## ====== at least one of them has valid pixels ===================================================  
                COMMON_DAYS_BLOCK_N = 0
                valid_year_doys_union[:] = False # to control the L8 and S2 are still alignment 
                max_n_year = 0
                for yi in range(N_YEAR_WINDOW):
                    max_n_year = 0
                    for i in range(PERIODS):
                        # If any L30 or S30 file in a given year at any given doy has valid pixels values (ref) 
                        if valid_year_doys[yi*PERIODS*2+i]>0 or valid_year_doys[yi*PERIODS*2+i+PERIODS]>0:
                            valid_year_doys_union[yi*PERIODS*2+i        ] = True # Landsat 08 DOY == True
                            valid_year_doys_union[yi*PERIODS*2+i+PERIODS] = True # Sentinel-2 DOY == True
                            max_n_year = max_n_year+1
                    
                    if COMMON_DAYS_BLOCK_N<max_n_year:
                        COMMON_DAYS_BLOCK_N=max_n_year
                
                ## ====== Only process those paried L30 and S30 files that either of them have valid pixels =======   
                process_data_i[:] = FILL # initialize this array as everytime this chunks needs to store new values
                for yi in range(N_YEAR_WINDOW):
                    images_with_valid = 0
                    for i in range(PERIODS):
                        # if any doy in a given year is true, which means that day has valid L30 or S30 data
                        if valid_year_doys_union[yi*PERIODS*2+i]: 
                            l8i = i+yi*PERIODS*2          # L30 file location i (0-PERIODS) that may have FILL values inside 
                            s2i = i+yi*PERIODS*2+PERIODS  # S30 file location i (PERIODS - 2*PERIODS) that may have FILL values inside 
                            l8newi = images_with_valid+(yi*COMMON_DAYS_BLOCK_N*2)  # L30 index that may have valid value                     
                            s2newi = images_with_valid+(yi*COMMON_DAYS_BLOCK_N*2)+COMMON_DAYS_BLOCK_N  # S30 index that may have valid value      
                            process_data_i[:process_n,l8newi,:BANDS_N] = all_data[valid_pixel_image,l8i,:] # read into L8 valids 
                            process_data_i[:process_n,s2newi,:BANDS_N] = all_data[valid_pixel_image,s2i,:] # read into S2 valids
                            common_index = np.logical_or (all_qa[valid_pixel_image,l8i], all_qa[valid_pixel_image,s2i]) # common locations that has at least one valid value
                            
                            l8_doy = np.unique (all_data[valid_pixel_image,l8i,0]) # the DOY at locations with valid reflectance for a given L30 file;  
                            s2_doy = np.unique (all_data[valid_pixel_image,s2i,0]) # the DOY at locations with valid reflectance for a given S30 file; 
                            l8s2_doy = np.unique (np.concatenate([l8_doy,s2_doy])) # only three cases, [-9999., DOY] or [DOY,-9999.] or [DOY, DOY]
                            l8s2_doy = l8s2_doy[l8s2_doy!=FILL]
                            
                            # Assign DOY to those pixel locations when only L30 or S30 has valid values;
                            # example: From :: S30 [1,3,4,...,FILL] L30 [1,FILL,4,...,95] 
                            #          To   :: S30 [1,3,4,...,95  ] L30 [1,3,   4,...,95]   
                            if l8s2_doy.size == 1: 
                                process_data_i[common_index,l8newi,0] = l8s2_doy[0] 
                                process_data_i[common_index,s2newi,0] = l8s2_doy[0] 
                                images_with_valid = images_with_valid+1 
                            else:
                                print ('!!!!!!!!!!!!!!!!!! l8s2_doy.size != 1:')
                                                
                # =========  degug  ======================================================================= 
                # aa = process_data_i[0,:,:]
                # aa_i = aa[:COMMON_DAYS_BLOCK_N,0] !=-9999. 
                # bb_i = aa[(2*COMMON_DAYS_BLOCK_N):(4*COMMON_DAYS_BLOCK_N),0] !=-9999. 
                # aa[:COMMON_DAYS_BLOCK_N,0][aa_i]
                # aa[(2*COMMON_DAYS_BLOCK_N):(4*COMMON_DAYS_BLOCK_N),0][bb_i]
                # aa[COMMON_DAYS_BLOCK_N:(2*COMMON_DAYS_BLOCK_N),1][aa_i]
                # aa[:COMMON_DAYS_BLOCK_N,0][aa_i] - aa[COMMON_DAYS_BLOCK_N:(2*COMMON_DAYS_BLOCK_N),0][aa_i]
                # aa[:COMMON_DAYS_BLOCK_N,1][aa_i] - aa[COMMON_DAYS_BLOCK_N:(2*COMMON_DAYS_BLOCK_N),1][aa_i]
                ## *****************************************************************************************
                
                ## classification & save image & browse 
                strategy = tf.distribute.MirroredStrategy()
                # BATCH = 1024 # 176 
                with strategy.scope():
                    model = load_model(model_path, COMMON_DAYS_BLOCK_N)
                    # add: , workers=4, use_multiprocessing=True 10/11/2024; didn't decrease the calculation time.
                    logits = model.predict(process_data_i[:process_n,:(COMMON_DAYS_BLOCK_N*N_YEAR_WINDOW*2)], batch_size=BATCH, verbose=2) 
            
                ## convert to CDL labels 
                inverse_mapping = np.load('inverse_mapping.npy')
                classes = np.argmax(logits,axis=-1).astype(np.uint8)
                class_image[outi:(outi+widthi),outj:(outj+heightj)][valid_pixel_image] = inverse_mapping[classes]
                tf.keras.backend.clear_session()
                gc.collect()
                
                # break
                       
            # break
        # class_image [np.logical_not(valid_pixel_image) ] = 255
    
        
        end = datetime.now()
        elapsed = end-start
        print_str = '\nEnd time = '+str(end) + 'Elapsed time = '+str(elapsed) + '\n======================================'
        print(print_str); 
        
        # class_image[class_image==6] = 7 ## this should be included on May 20, 2024 - NO NEED this as the legend has been adjusted to this 
        ## ============================================================================================================
        ## save result and browse 
        
        ## make month 9 to 09, month 1 to 01
        if DOYset == -1:
            name_header = "LC_{:4d}_".format(FIRST_YEAR+N_YEAR_WINDOW-1) + "Month_" + f"{Month:02}" + "_" + version + "_"
        else:
            name_header = "LC_{:4d}_".format(FIRST_YEAR+N_YEAR_WINDOW-1) + "DOY_" + f"{DOYset:03}" + "_" + version + "_"
            
        ## output the NRT predictions
        # name_header = "LC_{:4d}_".format(FIRST_YEAR+N_YEAR_WINDOW-1)
        HLSi.save_image_file (class_image[:,:],prefix=name_header,folder=CLASS_DIR)
        
        ## color display for selected HLS tiles
        if golden_tiles (tile_id):
          outname = BROWSE_DIR + name_header + tile_id + "_Transformer.tif"
          color_display.color_display_from_image(class_image[:,:], dsr_file="./CDL.dsr", output_tif= outname)
    
