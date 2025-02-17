# In-season crop type mapper
The python codes implement a within season crop type mapper for the conterminous United States using NASA Harmonized Landsat Sentinel-2 (HLS) (v2.0) data to produce maps for 37 crop types using a Transformer-based model trained using >1 million time series samples collected from 2016-2022 Cropland Data Layer (CDL) products. The mapper can be run on any date of the year. 

This mapper utilizes two years of HLS data (the previous year and the current year) and directly processes NASA HLS data as input without requiring any pre-processing. As more HLS data become available throughout the current year, the mapper enables within season crop type classifications with progressively improving classification accuracy. 

# The 37 crop types are: 
Corn, cotton, fallow/idle cropland, soybeans, winter wheat, barley, dbl/crop winwht/soybeans, potatoes, rice, sorghum, spring wheat, sugar beets, canola, dry beans, peanuts, peas, sunflowers, tomatoes, almonds, blueberries, citrus, grapes, olives, oranges, pecans, walnuts, cherries, millet, oats, other hay/non alfalfa, sweet corn, other crops.

# Usage
Pro_load_model_run_tile_v5_1.py [Input_dir] [Output_dir] [HLS_tile] [year] [day_of_year] [batch_size]

  Input_dir: The input HLS data directory, the HLS data are orgnized the same way as NASA or any other ways. The filename should not be chagned. It is better to end with '/' or '\'
  
  Output_dir: The output directory to store the map result, it is better to end with '/' or '\'
  
  HLS_tile: The HLS tile name, e.g., T14TPP - must start with 'T'
  
  Year: The current year for which the crop type mapping is generated
  
  Day of year: The day of year for which the crop type mapping is generated
  
  Batch size: The batch size used for the model running, default to 2048

The mean and stardard deviation file is located in this repository.  
The model and training samples are located in https://zenodo.org/records/14715402

# Requirements
Programming Languages: Python 3.8+
  
Libraries:
    tensorflow
    numpy
    rasterio
  
# Reference
More details can be refer to a paper: 
Zhang, H. K., Shen, Y., Zhang, X., Che, X., Yang, Z., et al. (2025), Robust and timely within season conterminous United States crop type mapping using the time series transformer architecture, In review. 

This Transformer-based model is based on the classifying the raw irregular time sereis (CRIT) model: 

"Zhang, H. K., Luo, D., & Li, Z. (2024). Classifying raw irregular time series (CRIT) for large area land cover mapping by adapting transformer model. Science of Remote Sensing, 100123. https://doi.org/10.1016/j.srs.2024.100123" with a set of customalization and adaptions due to different data sturctures and inputs. 

