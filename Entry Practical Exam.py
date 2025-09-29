# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:39:00 2025

@author: 52722
"""

# Librerías
from pathlib import Path
import pandas as pd
pip install wbgapi pandas
import wbgapi as wb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

#------------------------------------------------------------------------------
# Question 1. Comprehensive Data Acquisition and Preprocessing
#------------------------------------------------------------------------------
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
plt.savefig("D:/Vacante Tec de Monterrey/Entering-Practical-Exam-/Archivos para Pregunta 1/nas.png")
plt.show()     ## Para ver de forma visual las variables que se pueden descartar por 
               ## porcentaje de NAs.

            
# Descartamos tres variables, porque son mayores a 40%.
vars_keep = na_pct[na_pct < 40].index
df_panel_reduced = df_panel[vars_keep.to_list() + ["economy", "year"]]
df_panel_reduced = df_panel_reduced.reset_index(drop=True)
df_panel_reduced = df_panel_reduced.loc[:, ~df_panel_reduced.columns.duplicated()]
df_panel_reduced = df_panel_reduced.drop(columns=["level_0","index"])
df_panel_reduced.to_csv("D:/Vacante Tec de Monterrey/Entering-Practical-Exam-/Archivos para Pregunta 1/panel.csv")

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
descr.to_csv("D:/Vacante Tec de Monterrey/Entering-Practical-Exam-/Archivos para Pregunta 1/estadistica_descriptiva.csv")
# Correlaciones entre variables
num_vars = df_panel_reduced.drop(columns=["economy","year"])

corr = num_vars.corr(method="pearson")
corr.to_csv("D:/Vacante Tec de Monterrey/Entering-Practical-Exam-/Archivos para Pregunta 1/correlacion.csv")
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Matriz de correlaciones")
plt.tight_layout()
plt.savefig("D:/Vacante Tec de Monterrey/Entering-Practical-Exam-/Archivos para Pregunta 1/heatmap.png")
plt.show()   # Las variables que muestran una correlación mayor con las 
             # emisiones de CO2 son el PIB y el total de población.

#------------------------------------------------------------------------------
# Question 2. Predictibve Modeling and Scenario Analysis
#------------------------------------------------------------------------------

# Decidí trabajar un modelo log-log (para emisiones de CO2, GDP y Población)
df_model = df_panel_reduced.dropna(subset=["co2_total_mt", "gdp_usd"])
df_model = df_model.copy()
df_model["log_co2"] = np.log(df_model["co2_total_mt"])
df_model["log_gdp"] = np.log(df_model["gdp_usd"])
df_model["log_pop"] = np.log(df_model["population_total"])

X = df_model[["log_gdp", "log_pop", "renew_energy_pct", "fossil_energy_pct",
              "elec_use_kwh_pc", "urban_pop_pct"]]  # Variables explicativas
y = df_model["log_co2"]     # Variable explicada

# Para la selección de los conjuntos (training & testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Modelo de regresión múltiple Log-Log, como modelo de entrenamiento
print(np.any(np.isnan(X_train)))
print(np.any(np.isinf(X_train)))
rows_with_nan = X_train[X_train.isna().any(axis=1)]
print("NaN encontrados en:\n", rows_with_nan)
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
y_train = y_train.loc[X_train.index]  # sincronizar índices

X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
y_test = y_test.loc[X_test.index]

X_train_const = sm.add_constant(X_train)       # Entrenamiento
model = sm.OLS(y_train, X_train_const).fit()

print(model.summary())     # El coeficiente del log(GDP) es 0.4317

X_test_const = sm.add_constant(X_test)   # Prueba
y_pred = model.predict(X_test_const)

# Métricas
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² en test: {r2:.3f}")
print(f"RMSE en test: {rmse:.3f}")

#El modelo ajustado explica 0.4317% de la variación en emisiones en el conjunto de 
#prueba (R²). El error de predicción promedio es 0.418. El coeficiente de 
#PIB indica que un aumento de 10% en el PIB genera aproximadamente 4.317% de 
#aumento en emisiones, manteniendo las demás variables constantes.

# Se incluyen gráficos de validación
# Observado vs Predicho (test)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], linestyle="--")  # línea 45°
plt.xlabel("log(CO2) observado (test)")
plt.ylabel("log(CO2) predicho (test)")
plt.title("Validación: Observado vs Predicho (OLS)")
plt.tight_layout()
plt.savefig("D:/Vacante Tec de Monterrey/Entering-Practical-Exam-/Archivos para Pregunta 2/observado vs predicho.png")
plt.show()

# Residuales en test
resid = y_test - y_pred
plt.figure(figsize=(6,4))
plt.hist(resid, bins=30)
plt.xlabel("Residual (log CO2)")
plt.ylabel("Frecuencia")
plt.title("Distribución de residuales (test)")
plt.tight_layout()
plt.savefig("D:/Vacante Tec de Monterrey/Entering-Practical-Exam-/Archivos para Pregunta 2/residuales.png")
plt.show()

# Escenario simulando el aumento de 10% al PIB (GDP)
cols_X = ["log_gdp","log_pop","renew_energy_pct","fossil_energy_pct",
          "elec_use_kwh_pc","urban_pop_pct"]
need_pos = ["co2_total_mt","gdp_usd","population_total"]
df_ok = df_model.copy()
df_ok = df_ok[(df_ok[need_pos] > 0).all(axis=1)]

def latest_complete(g):
    g = g.sort_values("year")
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_X)
    if g.empty:
        return None
    return g.tail(1)

df_latest_list = []
for econ, g in df_ok.groupby("economy"):
    row = latest_complete(g)
    if row is not None:
        df_latest_list.append(row)

df_latest = pd.concat(df_latest_list, ignore_index=True)

X_base = df_latest[cols_X].copy()
X_cf = df_latest[cols_X].copy()

X_cf["log_gdp"] = X_cf["log_gdp"] + np.log(1.10)
X_base_c = sm.add_constant(X_base, has_constant='add')
X_cf_c = sm.add_constant(X_cf,   has_constant='add')

want_cols = list(model.params.index)     # ej. ['const','log_gdp', ...]
Xb_c = X_base_c.reindex(columns=want_cols)
Xc_c = X_cf_c.reindex(columns=want_cols)

mask_valid = ~Xb_c.isna().any(axis=1) & ~Xc_c.isna().any(axis=1)
Xb_c = Xb_c.loc[mask_valid]
Xc_c = Xc_c.loc[mask_valid]
df_latest = df_latest.loc[mask_valid]

assert set(Xb_c.columns) == set(want_cols)
assert not np.isnan(Xb_c.values).any()
assert not np.isinf(Xb_c.values).any()

yhat_base_log = model.predict(Xb_c)
yhat_cf_log = model.predict(Xc_c)

delta_log = yhat_cf_log - yhat_base_log                        # cambio en log
pct_change = (np.exp(delta_log) - 1.0) * 100                   # % exacto

sim_result = df_latest[["economy","year"]].copy()
sim_result["pct_change_co2_if_gdp_up_10"] = pct_change

# Resumen de la simulación
beta_gdp = model.params.get("log_gdp", np.nan)
elasticity_summary = {
    "beta_gdp": beta_gdp,
    "implied_pct_change": float(beta_gdp)*10 if pd.notna(beta_gdp) else np.nan,
    "mean_simulated_pct_change": float(sim_result["pct_change_co2_if_gdp_up_10"].mean())
}
print(sim_result.head())
print(elasticity_summary)

# Gráfico de los 10 países con mayor elasticidad relativa 
top10_pct = (sim_result
             .sort_values("pct_change_co2_if_gdp_up_10", ascending=False)
             .head(10)
             .sort_values("pct_change_co2_if_gdp_up_10"))  

plt.figure(figsize=(9,6))
plt.barh(top10_pct["economy"], top10_pct["pct_change_co2_if_gdp_up_10"])
plt.xlabel("% cambio en CO₂ (PIB +10%, ceteris paribus)")
plt.title("Top 10 países por cambio porcentual de CO₂ (modelo OLS log-log)")
for i, v in enumerate(top10_pct["pct_change_co2_if_gdp_up_10"]):
    plt.text(v, i, f"{v:.1f}%", va="center", ha="left")
plt.tight_layout()
plt.show()

# Análisis e interpretación de los resultados
beta = model.params["log_gdp"]
se = model.bse["log_gdp"]
g = np.log(1.10)

pct_point = (np.exp(beta*g) - 1) * 100
pct_low = (np.exp((beta - 1.96*se)*g) - 1) * 100
pct_high = (np.exp((beta + 1.96*se)*g) - 1) * 100

print(f"Efecto % (PIB +10%): {pct_point:.2f}%  (IC95%: {pct_low:.2f}%, {pct_high:.2f}%)")

# Para observar los cambios en términos absolutos
co2_b = np.exp(yhat_base_log)
co2_c = np.exp(yhat_cf_log)

sim_result["co2_pred_base_mt"] = co2_b
sim_result["co2_pred_cf_mt"]   = co2_c
sim_result["abs_change_mt"] = sim_result["co2_pred_cf_mt"] - sim_result["co2_pred_base_mt"]

q = sim_result["abs_change_mt"].quantile([0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]).round(2)
print("ΔCO2 (Mt) distribución — min, p10, p25, p50, p75, p90, max:")
print(q)

top5_up   = sim_result.nlargest(5, "abs_change_mt")[["economy","abs_change_mt"]]
top5_down = sim_result.nsmallest(5, "abs_change_mt")[["economy","abs_change_mt"]]
print("Top 5 mayor aumento (Mt):\n", top5_up)
print("Top 5 menor/negativo (Mt):\n", top5_down)
