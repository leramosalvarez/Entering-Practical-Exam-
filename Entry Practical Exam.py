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

# Question 1. Comprehensive Data Acquisition and Preprocessing

#data = wb.series.info(q='CO2')
#print(data)
variables = [                # Estas son las variables que se eligieron inicialmente
    # CO2
    "EN.GHG.CO2.AG.MT.CE.AR5",
    "EN.GHG.CO2.MT.CE.AR5",
    "EG.GDP.PUSE.KO.PP",    #GDP per unit of energy use (PPP $ per kg of oil equivalent)
    # Ambiental
    "EG.FEC.RNEW.ZS",  #Renewable energy consumption (% of total final energy consumption)
    "EG.USE.COMM.FO.ZS", #Fossil fuel energy consumption (% of total)
    "EG.USE.ELEC.KH.PC",  #Electric power consumption (kWh per capita)
    # Socioeconómicos
    "NY.GDP.MKTP.CD",     #GDP (current US$)
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

print(df_tidy.head())

df_panel = df_tidy.pivot_table(index=["economy","year"], columns="series", 
                               values="value").reset_index()

# Limpieza de datos (missing values, outliers y consistencia)
tidy.groupby("series")["value"].count().sort_values()
