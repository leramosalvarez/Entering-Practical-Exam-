# Pregunta 2 — Modelado predictivo y análisis de escenarios (CO₂ vs. PIB)

**Objetivo.** Construir un modelo predictivo de emisiones de CO₂ con indicadores socioeconómicos y ambientales y responder:  
**“Si un país incrementa su PIB 10%, ¿en cuánto cambian las emisiones de CO₂, manteniendo constantes los demás factores?”**

**Script base:** `Entry Practical Exam.py`

---

## 1) Elección del modelo

Se emplea una **regresión múltiple log–log (OLS)**:

\[
\ln(\text{CO₂})=\alpha+\beta\,\ln(\text{PIB})+\gamma'\mathbf{X}+\varepsilon
\]

- **Interpretación (elasticidad).** El coeficiente \(\beta\) es la **elasticidad** de CO₂ respecto al PIB. Un +10% en PIB implica un cambio porcentual aproximado de \(\beta\times 10\%\) y **exacto** de \(100\big((1.10)^{\beta}-1\big)\%\).
- **Controles \(\mathbf{X}\).** Incluyen (según disponibilidad en el panel): \(\ln(\text{Población})\), % renovables, % fósiles, consumo eléctrico per cápita, urbanización, ratio empleo/población. Estos capturan diferencias de tamaño, intensidad y mezcla energética entre países.
- **¿Por qué OLS log–log?.** Ofrece claridad al momento de la itnerpretación (elasticidades) y responde directamente la pregunta. 

---

## 2) Datos, limpieza y división entrenamiento/prueba

- **Estructura.** Panel país–año.
- **Limpieza clave.** Se filtran observaciones con \(\text{CO₂}, \text{PIB}, \text{Población}>0\) para evitar \(\log(0)\); se reemplazan \(\pm\infty\) por `NaN` y se eliminan filas incompletas en predictores.
- **Transformaciones.** \(\ln(\text{CO₂})\), \(\ln(\text{PIB})\), \(\ln(\text{Población})\).
- **Validación.** División **70/30** aleatoria (panel transversal):  
  - **Training (70%)**: estimación de coeficientes.  
  - **Testing (30%)**: evaluación fuera de muestra (**R²**, **RMSE**) sobre \(\ln(\text{CO₂})\).

> **Nota técnica:** Al predecir se fuerza la misma estructura de columnas (incluida la constante) usada en el entrenamiento para evitar `NaN` por desalineación.

---

## 3) Entrenamiento y validación

- **Modelo.** OLS con constante, sobre el conjunto de entrenamiento.  
- **Métricas (test).**  
  - **R² (test):** _0.973_  
  - **RMSE (test, en log):** _0.418_

**Gráficos de validación** (generados por el script):
- Observado vs. Predicho (línea 45°). (Archivo "observados vs predicho.png")
- Histograma de residuales (test).  (Archivo "residuales.png")

---

## 4) Análisis de escenario: **PIB +10%** (ceteris paribus)

**Procedimiento.**
1. Para cada país, se toma la **última observación con predictores completos**.  
2. Se construyen dos matrices:
   - **Baseline:** \(X\) observada.
   - **Contrafactual:** \(X\) con \(\ln(\text{PIB})\) aumentado en \(\ln(1.10)\) (subida exacta de 10% en PIB).
3. Se predice \(\hat{y}_{\text{base}}\) y \(\hat{y}_{\text{cf}}\) en log, se pasa a niveles y se calcula:
   - **Cambio % exacto:** \(100\left(e^{\hat{y}_{\text{cf}}-\hat{y}_{\text{base}}}-1\right)\).
   - **Cambio absoluto (Mt):** \(e^{\hat{y}_{\text{cf}}}-e^{\hat{y}_{\text{base}}}\).

**Propiedad del modelo log–log.** El **% de cambio** es prácticamente **común** a todos los países (depende de \(\beta\)); lo que sí varía entre países es el **cambio absoluto en Mt**, al depender del nivel base de emisiones.

---

## 5) Resultados e interpretación

### 5.1 Elasticidad PIB–CO₂
- **Coeficiente \(\beta\) de \(\ln(\text{PIB})\):** _0.4317_  
- **Cambio porcentual ante +10% en PIB:**
  - **Exacto:** \(100\big((1.10)^{\beta}-1\big)\% = \) _4.317380854631286_  
  - **Aprox. lineal:** \(\beta\times 10\% = \) _4.200738908024259_

### 5.2 Incertidumbre estadística (rango global)
- **IC95% de \(\beta\):** \([\beta-1.96\,se,\; \beta+1.96\,se]\)  
- **Rango esperado del % de cambio ante +10% en PIB:**  
  \[
  \left[\,100\big((1.10)^{\beta-1.96\,se}-1\big),\;100\big((1.10)^{\beta+1.96\,se}-1\big)\,\right]\% 
  \]
  = _3.99%, 4.41%_.

### 5.3 Heterogeneidad entre países (Mt)
- **Distribución de \(\Delta \text{CO₂}\) (Mt):** min, p10, p25, mediana, p75, p90, max = _0.03, 0.25, 0.48, 3.20, 24.81, 170.53, 1616.80_.

**Lectura ejecutiva.**  
“A un +10% de PIB, las emisiones aumentarían alrededor de **3.99-4.41% (IC95%)**. En términos absolutos, el cambio mediano es **3.20 Mt** (IQR **[0.48, 24.81] Mt**), con impactos máximos cercanos a **1616.80 Mt** en los países con mayor base de emisiones.”

---

## 6) Limitaciones y extensiones

- **Asociación, no causalidad.** Interpretación **ceteris paribus** dentro del modelo.  
- **Errores estándar.** Puede reportarse HC1 o **clustering por país** si se usan múltiples años por economía.  
- **Robustez.** Probar especificaciones alternativas (p. ej., interacciones \(\ln(\text{PIB})\times\) mezcla energética) y comparar contra **Random Forest/Boosting** para verificar estabilidad del efecto y mejorar poder predictivo.  
- **Versión per cápita.** Repetir con \(\ln(\text{CO₂ pc})\) para neutralizar tamaño poblacional.

---