# Real-time crop type mapper
The python codes implement a near real-time crop type mapper for the conterminous United States using NASA Harmonized Landsat Sentinel-2 (HLS) (v2.0) data to produce maps for 37 crop types using a Transformer-based model trained using >1 million time series samples collected from 2016-2022 Cropland Data Layer (CDL) products. 

# The 37 crop types are: 
Corn, cotton, fallow/idle cropland, soybeans, winter wheat, barley, dbl/crop winwht/soybeans, potatoes, rice, sorghum, spring wheat, sugar beets, canola, dry beans, peanuts, peas, sunflowers, tomatoes, almonds, blueberries, citrus, grapes, olives, oranges, pecans, walnuts, cherries, millet, oats, other hay/non alfalfa, sweet corn, other crops.

# Usage
This mapper utilizes two years of HLS data (the previous year and the current year) and directly processes NASA HLS data as input without requiring any pre-processing. The mapper can be run on any date of the year. As more HLS data become available throughout the current year, the mapper enables near real-time crop type classifications with progressively improving classification accuracy. 

Pro_load_model_run_tile_v5_0.py [Input_dir] [HLS-tile] [year] [day_of_year] [batch_size]
  Input_dir: The input HLS data directory, the HLS data are orgnized the same way as NASA or any other ways. The filename should not be chagned.  
  HLS-tile: the HLS tile name, e.g., T14TPP 
  Year: refers to the current year for which the crop type mapping is generated
  Day of year: refers to the day of year for which the crop type mapping is generated
  Batch size: the batch size used for the model running, default to 512

The model is where 
The mean and stardard deviations files are located in 
model. zip has all the necessary files including the trained model (.h5) and means/std files (.csv)

# Requirements
Programming Languages: Python 3.7+
  Libraries:
  tensorflow
  numpy
  rasterio
  
# reference
There is a paper in review for the codes: 

This Transformer-based model is based on the classifying the raw irregular time sereis (CRIT) model: "Zhang, H. K., Luo, D., & Li, Z. (2024). Classifying raw irregular time series (CRIT) for large area land cover mapping by adapting transformer model. Science of Remote Sensing, 100123. https://doi.org/10.1016/j.srs.2024.100123" with a set of customalization and adaptions due to different data sturctures and inputs. 

.

