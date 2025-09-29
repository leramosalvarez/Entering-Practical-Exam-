# Dataset Summary – CO₂ emissions & socioeconomic/environmental indicators

## Fuentes y alcance
- **Fuente**: World Bank Open Data (`wbgapi`).
- **Selección temporal**: últimos 35 años disponibles (`mrv=35`; es decir, el periodo es 1990–2024 según disponibilidad por variable y país).
- **Cobertura geográfica**: panel de países (todas las economías reportadas por el BM en las series seleccionadas).

## Variables incluidas (código → nombre corto)
**CO₂ emmissions**
- EN.GHG.CO2.AG.MT.CE.AR5 → `co2_agri_mt` (CO₂ de agricultura, Mt)
- EN.GHG.CO2.MT.CE.AR5 → `co2_total_mt` (CO₂ total, Mt; excluye LULUCF)
- EG.GDP.PUSE.KO.PP → `gdp_per_energy_ppp` (PIB por unidad de energía, PPP USD/kg oe)

**Ambiental / energía**
- EG.FEC.RNEW.ZS → `renew_energy_pct` (% renovables en consumo final)
- EG.USE.COMM.FO.ZS → `fossil_energy_pct` (% energía fósil)
- EG.USE.ELEC.KH.PC → `elec_use_kwh_pc` (kWh per cápita)

**Socioeconómicas**
- NY.GDP.MKTP.CD → `gdp_usd_current` (PIB US$ corrientes)
- NY.GDP.MKTP.KD → `gdp_usd` (PIB US$ constantes 2015)
- NY.GDP.DEFL.KD.ZG → `gdp_deflator_infl_pct` (inflación, % anual)
- SP.POP.GROW → `pop_growth_pct` (crec. poblacional, % anual)
- SP.POP.TOTL → `population_total` (población total)
- SE.XPD.TOTL.GD.ZS → `edu_exp_pct_gdp` (gasto educación, % PIB)
- SP.URB.TOTL.IN.ZS → `urban_pop_pct` (población urbana, %)
- SL.EMP.TOTL.SP.NE.ZS → `employment_pop_ratio` (empleo/población, %)

## Preprocesamiento aplicado
1. **Adquisición**: `wb.data.DataFrame(variables, mrv=35)` para obtener los últimos 35 años por variable/país.  
2. **Reshape**:  
   - `reset_index()` → `melt(economy, series → year, value)` → `year = int` (se quita el prefijo `YR`).  
   - `pivot_table(index=[economy, year], columns=series, values=value)` para formar el panel país–año.  
3. **Renombrado**: mapeo de códigos→nombres cortos (ver listado arriba).  
4. **Missingness**:  
   - Se calculó `% NA` por variable y se visualizó con un bar chart.  
   - **Criterio**: descartar series con **NA ≥ 40%**. En esta corrida fueron **3 variables** (las de mayor % NA).  
5. **Limpieza estructural**: se eliminaron columnas duplicadas y auxiliares (`level_0`, `index`) si aparecían tras el `pivot`.  

> Nota: la tabla de NAs y la figura (nas.png) sirven para justificar qué variables se retiran; el umbral del 40% es una regla práctica común para paneles internacionales.

## Estadísticas clave
- Dimensiones del panel (post-filtrado por NA): filas = *N_obs*, columnas = *N_vars* (incluye `economy` y `year`).  
- Periodo cubierto en los datos resultantes: **min(year)**–**max(year)** en el panel reducido.  
- Estadística descriptiva (media, desviación, min, max) calculada con `df.describe()`.  

## Correlaciones
- Se calculó la matriz de **Pearson** sobre variables numéricas (`drop(['economy','year'])`).  
- Hallazgo enfatizado: **CO₂ total** se correlaciona fuertemente con **PIB** y con **población total**.  
- **Figura**: heatmap de correlaciones.

## Observaciones / patrones
- Cobertura desigual entre series (p. ej., indicadores de energía/renovables tienen huecos más largos en los 90s).  
- Alta colinealidad esperable entre PIB, población, consumo eléctrico y CO₂ (importante para modelos lineales).  
- El filtrado por NA ≥ 40% reduce ruido de imputación y mejora consistencia para el modelado.

## Artefactos esperados
- `estadistica_descriptiva.csv` – resumen estadístico por variable.  
- `correlacion.csv` – matriz de correlaciones.  
- `heatmap.png` – mapa de calor de correlaciones.