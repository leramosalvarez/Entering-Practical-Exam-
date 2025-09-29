# Entering Practical Exam for the Decision Sciences Research Center at Tecnologico de Monterrey

Repositorio del **Practical Exam** para el Decision Sciences Research Center (Tec de Monterrey). Incluye preparación de datos, modelado econométrico y análisis de escenarios de CO₂ (PIB +10% y adopción EV 50%).

---

## 📁 Estructura

- [`Archivos para Pregunta 1`](./Archivos%20para%20Pregunta%201/) — limpieza, EDA y armado del panel.  
- [`Archivos para Pregunta 2`](./Archivos%20para%20Pregunta%202/) — modelo OLS log–log + escenario **PIB +10%**.  
- [`Archivos para Pregunta 3`](./Archivos%20para%20Pregunta%203/) — escenario **EV 50%**, sensibilidad y ranking de países.  
- [`Entry Practical Exam.py`](./Entry%20Practical%20Exam.py) — scrip.

> Usa rutas **relativas** dentro del repo para lectura/escritura en los scripts.

---

## ⚙️ Requisitos

```bash
python >= 3.10
pip install -r requirements.txt
# o bien:
pip install pandas numpy matplotlib statsmodels scikit-learn requests wbgapi
