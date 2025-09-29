# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:39:00 2025

@author: 52722
"""

# Paquetería
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
pip install wbgapi pandas
import wbgapi as wb
import matplotlib.pyplot as plt
import seaborn as sns

# Question 1. Comprehensive Data Acquisition and Preprocessing

data = wb.series.info(q='GDP')
print(data)
variables = [                # Estas son las variables que se eligieron inicialmente
    # CO2
    "EN.GHG.CO2.AG.MT.CE.AR5",  #Carbon dioxide (CO2) emissions from Agriculture (Mt CO2e)
    "EN.GHG.CO2.MT.CE.AR5",    #Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)
    "EG.GDP.PUSE.KO.PP",    #GDP per unit of energy use (PPP $ per kg of oil equivalent)
    # Ambiental
    "EG.FEC.RNEW.ZS",  #Renewable energy consumption (% of total final energy consumption)
    "EG.USE.COMM.FO.ZS", #Fossil fuel energy consumption (% of total)
    "EG.USE.ELEC.KH.PC",  #Electric power consumption (kWh per capita)
    # Socioeconómicos
    "NY.GDP.MKTP.CD",     #GDP (current US$)
    "NY.GDP.MKTP.KD",      #GDP (constant 2015 US$)
    "NY.GDP.DEFL.KD.ZG",   #Inflation, GDP deflator (annual %)
    "SP.POP.GROW",         #Population growth (annual %)
    "SP.POP.TOTL",         #Population, total
    "SE.XPD.TOTL.GD.ZS",   #Government expenditure on education, total (% of GDP)
    "SP.URB.TOTL.IN.ZS",   #Urban population (% of total population)
    "SL.EMP.TOTL.SP.NE.ZS",#Employment to population ratio, 15+, total (%) (national estimate)
]
df = wb.data.DataFrame(variables,mrv=35)

df_l = df.reset_index()

df_tidy = df_l.melt(
    id_vars=["economy", "series"], 
    var_name="year", 
    value_name="value"
)
df_tidy["year"] = df_tidy["year"].str.replace("YR","").astype(int)

print(df_tidy.info())

df_panel = df_tidy.pivot_table(index=["economy","year"], columns="series", 
                               values="value").reset_index()

colnames = {
    # CO2
    "EN.GHG.CO2.AG.MT.CE.AR5": "co2_agri_mt",        
    "EN.GHG.CO2.MT.CE.AR5": "co2_total_mt",          
    "EG.GDP.PUSE.KO.PP": "gdp_per_energy_ppp",       
    # Ambiental / Energía
    "EG.FEC.RNEW.ZS": "renew_energy_pct",            
    "EG.USE.COMM.FO.ZS": "fossil_energy_pct",        
    "EG.USE.ELEC.KH.PC": "elec_use_kwh_pc",          
    # Socioeconómicos
    "NY.GDP.MKTP.CD": "gdp_usd_current",             
    "NY.GDP.MKTP.KD": "gdp_usd",
    "NY.GDP.DEFL.KD.ZG": "gdp_deflator_infl_pct",    
    "SP.POP.GROW": "pop_growth_pct",                 
    "SP.POP.TOTL": "population_total",               
    "SE.XPD.TOTL.GD.ZS": "edu_exp_pct_gdp",          
    "SP.URB.TOTL.IN.ZS": "urban_pop_pct",            
    "SL.EMP.TOTL.SP.NE.ZS": "employment_pop_ratio",  
}

df_panel = df_panel.rename(columns=colnames)

# Limpieza de datos (missing values, outliers y consistencia)

na_count = df_panel.isna().sum()
na_pct = df_panel.isna().mean() * 100

na_summary = pd.DataFrame({
    "na_count": na_count,
    "na_pct": na_pct.round(2)
}).sort_values("na_pct", ascending=False)

print(na_summary)

na_summary["na_pct"].plot(kind="barh", figsize=(8,6))
plt.xlabel("% de valores faltantes")
plt.ylabel("Variables")
plt.axvline(x=40, color='red', linestyle='--', linewidth=2)
plt.axvline(x=20, color='black', linestyle='--', linewidth=2)
plt.show()     ## Para ver de forma visual las variables que se pueden descartar por 
               ## porcentaje de NAs.
               
# Descartamos tres variables, porque son mayores a 40%.
vars_keep = na_pct[na_pct < 40].index
df_panel_reduced = df_panel[vars_keep.to_list() + ["economy", "year"]]
df_panel_reduced = df_panel_reduced.reset_index(drop=True)
df_panel_reduced = df_panel_reduced.loc[:, ~df_panel_reduced.columns.duplicated()]
df_panel_reduced = df_panel_reduced.drop(columns=["level_0","index"])

# Resumen del conjunto de datos

# Dimensiones
print(f"Shape: {df_panel_reduced.shape}")  # filas, columnas
print("Columnas:", df_panel_reduced.columns.tolist())

# Rango temporal
print("Años:", df_panel_reduced["year"].min(), "-", df_panel_reduced["year"].max())

# Número de países
print("Países únicos:", df_panel_reduced["economy"].nunique())

# Estadística descriptiva
descr = df_panel_reduced.describe().T[["mean","std","min","max"]]
descr["missing_pct"] = df_panel_reduced.isna().mean()*100
print(descr)

# Correlaciones entre variables
num_vars = df_panel_reduced.drop(columns=["economy","year"])

corr = num_vars.corr(method="pearson")

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Matriz de correlaciones")
plt.tight_layout()
#plt.savefig("data/processed/correlation_heatmap.png")
plt.show()   # Las variables que muestran una correlación mayor con las 
             # emisiones de CO2 son el PIB y el total de población.




