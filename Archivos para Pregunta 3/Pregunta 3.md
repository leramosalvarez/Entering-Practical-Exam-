# Pregunta 3 — Escenario de un aumento de 50% en la adopción de vehículo eléctricos, sensibilidad y ranking de países

**Objetivo.** Estimar el impacto en las **emisiones globales de CO₂** si el **50% del stock** de autos de pasajeros pasa a vehículos eléctricos (EV), realizar **análisis de sensibilidad** (LOW/BASE/HIGH) y **identificar** los países con las reducciones más significativas, cuantificando el efecto.

---

## 1) Resumen metodológico

### 1.1. Definición del escenario
- Para cada país, se eleva la **adopción de EV (stock)** a **0.50** (50%).  
- Se mantiene **ceteris paribus** el resto de variables.

### 1.2. Cálculo
Para país \(i\), en el último año con datos completos:

- **Ahorro por combustión evitada**  
  \[
  \Delta E^{ICE}_i \;=\; s_{\text{passenger}}\;\times\;E_i\;\times\;\Delta \text{adopción}_i
  \]
  donde \(E_i\) son emisiones totales y \(s_{\text{passenger}}\) es la fracción del CO₂ atribuible a **autos de pasajeros**.

- **Emisiones añadidas por electricidad de EV**  
  \[
  \Delta E^{EV}_i \;=\; \frac{\text{Pob}_i \times \Delta \text{adopción}_i \times \text{km/año} \times \text{kWh/km} \times \text{gCO₂/kWh}_i}{10^{12}}
  \]
  (g→Mt: \(10^{12}\)).

- **Impacto neto**  
  \[
  \Delta E^{net}_i \;=\; \Delta E^{EV}_i \;-\; \Delta E^{ICE}_i
  \]
  Valor **negativo** ⇒ **reducción** de CO₂.

### 1.3. Sensibilidad (LOW/BASE/HIGH)
Parámetros posibles:

| Escenario | km/año | kWh/km | \(s_{\text{passenger}}\) | Multiplicador de red (\(m_{\text{grid}}\)) |
|---|---:|---:|---:|---:|
| **LOW**  | 14 000 | 0.22 | 0.08 | 1.15 |
| **BASE** | 10 000 | 0.18 | 0.10 | 1.00 |
| **HIGH** |  6 000 | 0.15 | 0.12 | 0.85 |

El factor \(m_{\text{grid}}\) aproxima que la **intensidad marginal** puede diferir de la media.

---

## 2) Datos y variables

- **Base P2** unificada con **mix eléctrico** (WDI) y **adopción EV (stock)** (OWID/IEA).  
- Transformaciones en el script:
  - `EG.ELC.FOSL.ZS` → `elec_gen_fossil_pct` (fósil % de generación eléctrica).  
  - `EG.ELC.RNWX.ZS` → `elec_gen_renew_pct`.  
  - EV stock share → `ev_stock_share_prop` (0–1) y `ev_stock_share_pct` (0–100).

- **Intensidad de red (gCO₂/kWh)**  
  - Si falta en datos, el script crea `grid_gco2_per_kwh` como **proxy** con el mix:  
    \[
    \text{gCO₂/kWh} \approx 900\times\text{fósil} + 50\times(1-\text{fósil})
    \]
  - Se aplica \(m_{\text{grid}}\) en la simulación para sensibilidad marginal.

---

## 3) Implementación (resumen del pipeline de código)

1. **Construcción del panel ampliado** (`df_merged`): merge por `economy`–`year` entre la base de P2 y la base de P3 (mix y EV).  
2. **Último año con datos completos** por país (`latest_complete`).  
3. **Simulación** con `simulate_ev50(...)`:
   - Calcula `delta_adopt` = \(\max(0.5 - \text{ev\_stock\_share},\,0)\).  
   - Computa `ice_avoided_mt`, `ev_added_mt`, `delta_net_mt`, `%` relativo.  
4. **Sensibilidad**: corre escenarios LOW/BASE/HIGH y genera:
   - **Resumen global** (`global_summary`): \(\sum \Delta E^{net}\) y % sobre el total.  
   - **Ranking** `top10_reduction` (reducción absoluta en Mt) y `bottom10_increase`.
5. **Identificación** de países con mayor reducción (BASE) y **revisar** de resultados/figuras. (archivos: top10_reduction_absolute_BASE.csv, top10_reduction_percent_BASE.csv, top10.png)

---

## 4) Supuestos y limitaciones

- Escenario **estático** (no dinámico) y **operacional**.  
- Parámetros uniformes por escenario (km/año, kWh/km) y fracción de **pasajeros** aproximada (\(s_{\text{passenger}}\)).  
- **Intensidad de red**: promedio ajustado por multiplicador marginal; la realidad horaria puede diferir.  
- **Ceteris paribus**: no se modela *rebound*, cambios de precios, despacho, pérdidas de carga adicionales, etc.

---

