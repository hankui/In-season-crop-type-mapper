# Real-time-crop-type-mapper
Python codes for CONUS near real-time mapper using HLS data

This codes implement a near real-time crop type mapper for the conterminous United States using NASA Harmonized Landsat Sentinel-2 (HLS) data to produce maps for 37 crop types using a Transformer-based model trained using >1 million time series samples collected from 2016-2022 Cropland Data Layer (CDL) products.

# The 37 crop types are: 

NOTE: This mapper is a two-year HLS data trained Transofrmer model, which means you need to prepare at least one-year-long HLS data. With more available HLS data in the current year (the following year of the year for the available HLS data), you can obtain near real-time crop type classifications based on this mapper with increasing crop classfication accuracy. 

Pro_load_model_run_tile_v5_0.py :: Use the trained Transformer model for crop type mapping at any given months or Day of Year (DOY) using HLS version 2.0 data.


There is a paper in review for the codes: 

This Transformer-based model is based on the classifying the raw irregular time sereis (CRIT) model: "Zhang, H. K., Luo, D., & Li, Z. (2024). Classifying raw irregular time series (CRIT) for large area land cover mapping by adapting transformer model. Science of Remote Sensing, 100123. https://doi.org/10.1016/j.srs.2024.100123" with a set of customalization and adaptions due to different data sturctures and inputs. 


model. zip has all the necessary files including the trained model (.h5) and means/std files (.csv).

