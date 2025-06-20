# WGU_D499_Project_2_Unsupervised_ML

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

---

## Identify Customer Segments — WGU D499 Capstone — D. Cook

This project applies dimensionality reduction (PCA) and clustering (KMeans) to analyze customer demographics for a German mail-order sales company.

The goal is to identify which customer segments are most overrepresented or underrepresented in the company's customer base — and explore what characteristics define those groups.

The project compares customer clusters to the general population of Germany, using unsupervised machine learning.

---

## Project Origin

This project was completed as part of the **WGU D499 Data Analytics Capstone** course.

The dataset and project prompt were provided via the WGU / Udacity coursework.

--- 

## Key Skills Demonstrated

✅ Data wrangling & cleaning  
✅ Handling missing data  
✅ Dimensionality reduction (PCA)  
✅ Clustering (KMeans)  
✅ Feature engineering  
✅ Data visualization (matplotlib, seaborn)  
✅ Portfolio-safe synthetic data generation  
✅ Reusable pipelines for processing customer and population data

---

## Project Structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         WGU_D499_P2_DCook and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── WGU_D499_P2_DCook   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes WGU_D499_P2_DCook a Python module
    │
    ├── clustering.py           <- Code to perform clustering of the data
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── eda.py                  <- Code to perform Exploratory Data Analysis 
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── impute_and_scale.py     <- Code to perform imputing and scaling of the data
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    |
    ├── pca.py                  <- Code to peform PCA functions
    |
    ├── plots.py                <- Code to create visualizations
    |
    ├── transformations.py      <- Code to perform transformation of the data
    │
    └── utils.py                <- Code to perform common helper functional tasks
```
---

## How to Run

1️⃣ Open the "WGU_D499_P2_DCook" folder"
2️⃣ Locate the "LaunchVEnvAndLabServer.ps1" file within the project folder
3️⃣ Right-Click on the file and select "Run with Powershell" from the context menu.


---

## Notes

- This project demonstrates end-to-end unsupervised learning on a realistic customer dataset.
- It is not intended to represent a real company — data is anonymized and synthetic for public use.
- All code, analysis, and write-up were completed by D. Cook as part of the WGU D499 capstone.

---



--------

