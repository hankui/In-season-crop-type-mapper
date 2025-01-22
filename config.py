# config
MAX_LANDSAT   = 176
MAX_SENTINEL2 = 176
YEARS_DATA = 2;

lst = [] ### it is actually the total number of Landsat and Sentinel-2 available bands (Total spectral bands #). 09132024 Yu Shen
L8_fields = []
S2_fields = []

# L8_fields.append('year')
for i in range(MAX_LANDSAT):
    lst.append(f'L{i:03d}.coastal')
    lst.append(f'L{i:03d}.blue')
    lst.append(f'L{i:03d}.green')
    lst.append(f'L{i:03d}.red')
    lst.append(f'L{i:03d}.nir')
    lst.append(f'L{i:03d}.swir1')
    lst.append(f'L{i:03d}.swir2')
    lst.append(f'L{i:03d}.bt1')
    lst.append(f'L{i:03d}.bt2')
    lst.append(f'L{i:03d}.doy')
    lst.append(f'L{i:03d}.qa')
    
    L8_fields.append(f'L{i:03d}.doy'    )
    L8_fields.append(f'L{i:03d}.coastal')
    L8_fields.append(f'L{i:03d}.blue'   )
    L8_fields.append(f'L{i:03d}.green'  )
    L8_fields.append(f'L{i:03d}.red'    )
    L8_fields.append(f'L{i:03d}.nir'    )
    L8_fields.append(f'L{i:03d}.swir1'  )
    L8_fields.append(f'L{i:03d}.swir2'  )

# S2_fields.append('year')
START_I_REF = 1 
for i in range(MAX_SENTINEL2):
    lst.append(f'S{i:03d}.coastal')
    lst.append(f'S{i:03d}.blue')
    lst.append(f'S{i:03d}.green')
    lst.append(f'S{i:03d}.red')
    lst.append(f'S{i:03d}.edge1')
    lst.append(f'S{i:03d}.edge2')
    lst.append(f'S{i:03d}.edge3')
    lst.append(f'S{i:03d}.nir8')
    lst.append(f'S{i:03d}.nirA')
    lst.append(f'S{i:03d}.swir1')
    lst.append(f'S{i:03d}.swir2')
    lst.append(f'S{i:03d}.doy')
    lst.append(f'S{i:03d}.qa')
    
    S2_fields.append(f'S{i:03d}.doy'    )
    S2_fields.append(f'S{i:03d}.coastal')
    S2_fields.append(f'S{i:03d}.blue'   )
    S2_fields.append(f'S{i:03d}.green'  )
    S2_fields.append(f'S{i:03d}.red'    )
    S2_fields.append(f'S{i:03d}.nirA'   )
    S2_fields.append(f'S{i:03d}.swir1'  )
    S2_fields.append(f'S{i:03d}.swir2'  )
    S2_fields.append(f'S{i:03d}.edge1'  )
    S2_fields.append(f'S{i:03d}.edge2'  )
    S2_fields.append(f'S{i:03d}.edge3'  )
    S2_fields.append(f'S{i:03d}.nir8'   )


# Provided mapping
mapping = {
    152: 'Shrubland', 176: 'Grass/Pasture', 142: 'Evergreen Forest', 141: 'Deciduous Forest', 1: 'Corn', 5: 'Soybeans', 
    190: 'Woody Wetlands', 121: 'Developed/Open Space', 143: 'Mixed Forest', 111: 'Open Water', 122: 'Developed/Low Intensity', 
    24: 'Winter Wheat', 37: 'Other Hay/Non Alfalfa', 195: 'Herbaceous Wetlands', 61: 'Fallow/Idle Cropland', 36: 'Alfalfa', 
    123: 'Developed/Med Intensity', 131: 'Barren', 23: 'Spring Wheat', 2: 'Cotton', 4: 'Sorghum', 124: 'Developed/High Intensity', 
    3: 'Rice', 26: 'Dbl Crop WinWht/Soybeans', 21: 'Barley', 28: 'Oats', 42: 'Dry Beans', 6: 'Sunflower', 75: 'Almonds', 
    41: 'Sugarbeets', 59: 'Sod/Grass Seed', 10: 'Peanuts', 31: 'Canola', 212: 'Oranges', 76: 'Walnuts', 53: 'Peas', 43: 'Potatoes', 
    22: 'Durum Wheat', 69: 'Grapes', 242: 'Blueberries', 29: 'Millet', 12: 'Sweet Corn', 68: 'Apples', 71: 'Other Tree Crops', 
    27: 'Rye', 66: 'Cherries', 205: 'Triticale', 72: 'Citrus', 54: 'Tomatoes', 74: 'Pecans', 236: 'Dbl Crop WinWht/Sorghum', 
    225: 'Dbl Crop WinWht/Corn', 32: 'Flaxseed', 33: 'Safflower', 52: 'Lentils', 254: 'Dbl Crop Barley/Soybeans', 11: 'Tobacco', 
    211: 'Olives', 13: 'Pop or Orn Corn', 44: 'Other Crops', 70: 'Christmas Trees', 238: 'Dbl Crop WinWht/Cotton', 58: 'Clover/Wildflowers', 
    204: 'Pistachios', 220: 'Plums', 49: 'Onions', 67: 'Peaches', 57: 'Herbs', 215: 'Avocados', 112: 'Perennial Ice/Snow', 
    237: 'Dbl Crop Barley/Corn', 46: 'Sweet Potatoes', 50: 'Cucumbers', 51: 'Chick Peas', 77: 'Pears', 228: 'Dbl Crop Triticale/Corn', 
    206: 'Carrots', 222: 'Squash', 48: 'Watermelons', 47: 'Misc Vegs & Fruits', 14: 'Mint', 210: 'Prunes', 229: 'Pumpkins', 56: 'Hops', 
    92: 'Aquaculture', 216: 'Peppers', 240: 'Dbl Crop Soybeans/Oats', 35: 'Mustard', 241: 'Dbl Crop Corn/Soybeans', 45: 'Sugarcane', 
    209: 'Cantaloupes', 39: 'Buckwheat', 213: 'Honeydew Melons', 243: 'Cabbage', 219: 'Greens', 207: 'Asparagus', 208: 'Garlic', 
    214: 'Broccoli', 226: 'Dbl Crop Oats/Corn', 217: 'Pomegranates', 250: 'Cranberries', 221: 'Strawberries', 246: 'Radishes', 
    60: 'Switchgrass', 227: 'Lettuce', 38: 'Camelina', 247: 'Turnips', 30: 'Speltz', 224: 'Vetch', 235: 'Dbl Crop Barley/Sorghum', 
    239: 'Dbl Crop Soybeans/Cotton', 244: 'Cauliflower', 245: 'Celery', 55: 'Caneberries', 218: 'Nectarines', 25: 'Other Small Grains', 
    34: 'Rape Seed', 248: 'Eggplants'
}