Main Folder: `INDmetv1.0`
This folder contains grid-wise daily and monthly precipitation, maximum temperature, and minimum temperature data in both CSV and NetCDF formats. In addition to gridded data, the dataset has been aggregated over different administrative boundaries — States, Districts, and Talukas.

The folder includes the following five sub-folders:

1. `INDmet_Taluka_Data`
2. `INDmet_State_Data`
3. `INDmet_District_Data`
4. `INDmet_Gridded_Data`
5. `INDmet_Netcdf_Data`

---

### Folder: `INDmet_Taluka_Data`

This folder includes taluka-wise daily and monthly precipitation and temperature data in CSV format. It contains two sub-folders:

1. `Taluka_data_Daily_CSV`
   Contains daily data for each taluka.

   * Filename format: `data_Taluka_<ID>.csv`
   * ID source: Refer to `India_Taluka.csv`
   * Columns (7):
     `Year`, `Month`, `Day`, `Precipitation (mm/day)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

2. `Taluka_data_Monthly_CSV`
   Contains monthly data for each taluka.

   * Filename format: `data_Taluka_<ID>.csv`
   * ID source: Refer to `India_Taluka.csv`
   * Columns (6):
     `Year`, `Month`, `Precipitation (mm/month)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

---

### Folder: `INDmet_State_Data`

This folder includes state-wise daily and monthly data. It contains two sub-folders:

1. `State_data_Daily_CSV`

   * Filename format: `data_State_<ID>.csv`
   * ID source: Refer to `India_States.csv`
   * Columns (7):
     `Year`, `Month`, `Day`, `Precipitation (mm/day)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

2. `State_data_Monthly_CSV`

   * Filename format: `data_State_<ID>.csv`
   * ID source: Refer to `India_States.csv`
   * Columns (6):
     `Year`, `Month`, `Precipitation (mm/month)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

---

### Folder: `INDmet_District_Data`

This folder includes district-wise daily and monthly data. It contains two sub-folders:

1. `District_data_Daily_CSV`

   * Filename format: `data_District_<ID>.csv`
   * ID source: Refer to `India_Districts.csv`
   * Columns (7):
     `Year`, `Month`, `Day`, `Precipitation (mm/day)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

2. `District_data_Monthly_CSV`

   * Filename format: `data_District_<ID>.csv`
   * ID source: Refer to `India_Districts.csv`
   * Columns (6):
     `Year`, `Month`, `Precipitation (mm/month)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

---

### Folder: `INDmet_Gridded_Data`

This folder includes grid-wise precipitation and temperature data for 0.05° × 0.05° spatial resolution grids across India. It contains two sub-folders:

1. `Gridded_Data_Daily_CSV`
   Contains daily data for each grid point.

   * Filename format: `data_<latitude>_<longitude>.csv`
   * Coordinates source: Refer to columns 1 and 2 of `INDmet_lalo_5km.txt`
   * Columns (7):
     `Year`, `Month`, `Day`, `Precipitation (mm/day)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

2. `Gridded_Data_Monthly_CSV`
   Contains monthly data for each grid point.

   * Filename format: `data_<latitude>_<longitude>.csv`
   * Coordinates source: Refer to columns 1 and 2 of `INDmet_lalo_5km.txt`
   * Columns (6):
     `Year`, `Month`, `Precipitation (mm/month)`, `Max_Temperature (°C)`, `Min_Temperature (°C)`, `Mean_Temperature (°C)`

---

### Folder: `INDmet_Netcdf_Data`

This folder includes NetCDF files of gridded daily data for the entire country at 0.05° spatial resolution. It contains four sub-folders:

1. `Yearly_File_Precipitation`

   * Daily precipitation files
   * Filename format: `INDmet_precipitation_05km_<year>.nc`

2. `Yearly_File_Tmax`

   * Daily maximum temperature files
   * Filename format: `INDmet_tmax_05km_<year>.nc`

3. `Yearly_File_Tmin`

   * Daily minimum temperature files
   * Filename format: `INDmet_tmin_05km_<year>.nc`

4. `Yearly_File_Tmean`

   * Daily mean temperature files
   * Filename format: `INDmet_tmean_05km_<year>.nc`

---
