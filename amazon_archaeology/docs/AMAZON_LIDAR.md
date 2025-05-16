# Specialized Amazon LiDAR Datasets

This document provides information about specialized LiDAR datasets covering the Amazon rainforest, particularly in Brazil. These datasets were collected through various research projects and contain valuable high-resolution elevation data that can be used to identify archaeological features.

## Available Datasets

### 1. Sustainable Landscapes Brazil (2008-2018)

**Dataset ID:** `ornl_slb_2008_2018`  
**URL:** [https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1644](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1644)  
**Description:** LiDAR point cloud data collected during surveys over selected forest research sites across the Amazon rainforest in Brazil between 2008 and 2018 for the Sustainable Landscapes Brazil Project.  
**Data Type:** Point cloud data  
**Coverage:** Multiple regions across Para, Amazonas, and Mato Grosso states  

**Key Regions:**
- Para: (-55.0, -5.0) to (-46.0, -1.0)
- Amazonas: (-62.0, -3.0) to (-58.0, 1.0)
- Mato Grosso: (-59.0, -12.0) to (-54.0, -7.0)

### 2. LiDAR Data over Manaus, Brazil (2008)

**Dataset ID:** `ornl_manaus_2008`  
**URL:** [https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1515](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1515)  
**Description:** LiDAR point clouds and digital terrain models (DTM) from surveys over the K34 tower site in the Cuieiras Biological Reserve, over forest inventory plots in the Adolpho Ducke Forest Reserve, and over sites of the Biological Dynamics of Forest Fragments Project (BDFFP) near Manaus.  
**Data Type:** Point cloud data and digital terrain models  
**Coverage:** Manaus region, Amazonas state  

**Key Regions:**
- K34 Tower: (-60.21, -2.61) to (-60.20, -2.60)
- Ducke Forest: (-59.98, -2.96) to (-59.91, -2.94)
- BDFFP: (-60.11, -2.41) to (-59.84, -2.38)

### 3. LiDAR Data over Paragominas, Brazil (2012-2014)

**Dataset ID:** `ornl_paragominas_2012_2014`  
**URL:** [https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1302](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1302)  
**Description:** Raw LiDAR point cloud data and derived Digital Terrain Models (DTMs) for five forested areas in the municipality of Paragominas, Para, Brazil, for the years 2012, 2013, and 2014.  
**Data Type:** Point cloud data and digital terrain models  
**Coverage:** Paragominas municipality, Para state  

**Key Regions:**
- Paragominas: (-48.0, -4.0) to (-46.5, -2.5)

### 4. LiDAR Transects across Brazilian Amazon

**Dataset ID:** `zenodo_brazilian_amazon`  
**URL:** [https://zenodo.org/records/7689909](https://zenodo.org/records/7689909)  
**Description:** LiDAR transects across the Brazilian Amazon, particularly Acre and Rondônia states, with randomly distributed transects over forest and secondary forest areas, coverage of the deforestation arch, and transects overlapping field plots for model calibration.  
**Data Type:** Point cloud data  
**Coverage:** Primarily Acre and Rondônia states  

**Key Regions:**
- Acre: (-73.0, -11.0) to (-66.7, -7.0)
- Rondonia: (-66.7, -11.0) to (-59.8, -7.0)

### 5. EMBRAPA Paisagens LiDAR

**Dataset ID:** `embrapa_paisagens_lidar`  
**URL:** [https://www.paisagenslidar.cnptia.embrapa.br/](https://www.paisagenslidar.cnptia.embrapa.br/)  
**Description:** Interactive map displaying Lidar and Forest Inventory data for Brazilian states, supporting research on sustainable landscapes and forest ecosystems.  
**Data Type:** Interactive map with LiDAR data  
**Coverage:** Various regions across Brazil  

### 6. Canopy Height Models from LiDAR

**Dataset ID:** `zenodo_canopy_height`  
**URL:** [https://zenodo.org/records/7104044](https://zenodo.org/records/7104044)  
**Description:** Canopy height models derived from LiDAR data collected across the Brazilian Amazon.  
**Data Type:** Canopy height models  
**Coverage:** Brazilian Amazon  

## Using Amazon LiDAR Datasets

### Listing Available Datasets

To see all available Amazon LiDAR datasets:

```bash
python run.py list-amazon-datasets
```

### Analyzing with a Specific Dataset

To run analysis using a specific Amazon LiDAR dataset:

```bash
python run.py analyze --amazon-lidar ornl_slb_2008_2018 --bounds=-55.0,-5.0,-54.0,-4.0
```

You can also specify a particular region within a dataset:

```bash
python run.py analyze --amazon-lidar ornl_manaus_2008 --amazon-region "K34 Tower"
```

### Data Access

Most of these datasets require manual registration and download from their respective repositories. When you specify an Amazon LiDAR dataset, the tool will generate a guidance file with instructions on how to access the data.

The downloaded data should be placed in the `amazon_archaeology/data/lidar/` directory for the tool to use it during analysis.

## Data Formats

The Amazon LiDAR datasets are typically available in the following formats:

1. **LAS/LAZ Files**: Point cloud data in LAS format (or compressed LAZ format)
2. **GeoTIFF**: Digital terrain models (DTM) and digital surface models (DSM)
3. **ASCII**: Point cloud data in text format

## Archaeological Value

These high-resolution LiDAR datasets are valuable for archaeological research in the Amazon for several reasons:

1. **Penetrating Forest Canopy**: LiDAR can penetrate dense forest canopy to reveal the ground surface beneath.
2. **Detecting Subtle Features**: The high resolution allows detection of subtle earthworks and geometric formations.
3. **Revealing Patterns**: Large-scale surveys can reveal patterns of settlement and land use.
4. **Providing Ground Truth**: These datasets can serve as ground truth for verification of features detected in other remote sensing data.

## Data Citation

When using these datasets, please cite the original data sources as specified in their respective repository documentation. 