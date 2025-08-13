🌿 AgroIntelligence: A Smart Crop Recommendation System
Welcome to the AgroIntelligence project! This repository contains a powerful, data-driven system for providing crop recommendations and farm guidance to farmers in Andhra Pradesh. By leveraging a comprehensive dataset and a machine learning model, this tool aims to optimize crop selection and improve agricultural outcomes.

🎥 Project Preview
Get a quick look at the application in action! This video demonstrates the functionality of the model and the user interface.

Watch the Preview Video Here: "https://drive.google.com/file/d/1SWAR2zwdrlNJhomP7i3jvbsgOn1Zywdn/view?usp=sharing"


📊 AP Crop Dataset (Deterministic, Realistic)
This project is built on the apcrop_dataset_realistic.csv file, a rich dataset with 18,240 rows of simulated agricultural data.

File: apcrop_dataset_realistic.csv
Rows: 18240

Column Descriptions:

Year: Year (2015−2024)

District: Andhra Pradesh district (26 districts)

Mandal: Deterministic mandal name placeholder (District_Mandal_1..N)

Season: Kharif / Rabi / Zaid

Soil_Type: Soil class used (Alluvial, Black, Red-Sandy, Mixed)

Soil_pH: Soil pH (deterministic district-level average)

Organic_Carbon_pct: Soil organic carbon percent

Soil_N_kg_ha, Soil_P_kg_ha, Soil_K_kg_ha: Deterministic soil nutrient levels (kg/ha)

Avg_Temp_C: Seasonal average temperature (NASA POWER derived)

Seasonal_Rainfall_mm: Seasonal precipitation (NASA POWER derived)

Avg_Humidity_pct: Seasonal average relative humidity

Water_Source: Deterministic water source for district/mandal

Previous_Crop: Deterministic previous crop (cycled)

Primary_Crop: First recommended crop (deterministic)

Secondary_Crop: Second recommended crop (deterministic)

Suitable_Crops: JSON array of suitable crops (Primary + Secondary)

Fertilizer_Plan: JSON dict with nutrient deficits and fertilizer (kg/ha) and a split schedule

Irrigation_Plan: JSON dict with seasonal (mm) requirement and suggestion for method and (mm/week)

Market_Price_Index: Deterministic market index per primary crop (0−1 scale) (fallback values used if Agmarknet unavailable)

⚠️ Assumptions & Limitations
Soil Data: Soil district values are deterministic estimates based on public SHC/ICAR summaries. For field-level precision, use lab soil tests.

Climate Data: NASA POWER provides satellite-based gridded climate; used here as a robust programmatic source for seasonal averages.

Market Data: Market prices are derived via fallback deterministic indices. For real-time pricing, integrate Agmarknet or mandi feeds.

Recommendations: Suitable crops mapping is rule-based agronomy guidance; you should refine with local KVK/Krishi advisories for specific mandals.

Dataset: This dataset is intentionally noise-free for reproducibility. If you want controlled variability for training robustness, enable the script's randomization option.

🛠️ How to Re-run
To regenerate this dataset and the associated files, follow these steps:

Install Requirements: pip install pandas requests tqdm

Run the Script: python make_apcrop_dataset.py

Outputs: apcrop_dataset_realistic.csv, sources.txt, README.md
