
## Installation Entorno local:
```
python3 -m venv venv  
source venv/bin/activate
pip install -r requirements.txt
docker build -t churn-model . (open docker app if needed)
docker info
```

##  Entrenamiento del modelo
Ejecuta este script para cargar los datos, preprocesarlos, entrenar y guardar el modelo:
Esto generará `model/model.pkl` y mostrará métricas como Accuracy, Recall, ROC AUC, etc.

```bash
python run_training.py
```
Loading and merging data...
Training logistic regression model...
Evaluating model...
{'Accuracy': 0.9202816982711346, 'Precision': 0.575, 'Recall': 0.008309248554913294, 'F1 Score': 0.01638176638176638, 'ROC AUC': np.float64(0.5038879910706434), 'Confusion Matrix': array([[31862,    17],
       [ 2745,    23]])}
Saving model...
All done.


## Inferencia por lotes (batch)

1. Asegúrate de tener un archivo CSV de entrada con las mismas columnas que las usadas en entrenamiento, excepto la columna `target`.

2. Ejecuta:
```bash
python batch_inference.py
```

Esto leerá `data/input_data.csv`, aplicará el modelo y guardará las predicciones en `output_predictions.csv`.

---

## 🐳 Ejecución con Docker

1. Construye la imagen:
```bash
docker build -t churn-model .
```

2. Ejecuta el contenedor:
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/model:/app/model" \
  churn-model
```

Esto correrá `batch_inference.py` dentro del contenedor y generará el archivo `output_predictions.csv`.

---

##  Notas finales
- El modelo espera exactamente las mismas columnas de entrada que se usaron para entrenarlo.
- Asegúrate de ejecutar `run_training.py` antes de correr `batch_inference.py`, o de tener `model.pkl` ya disponible.

¡Proyecto listo para entrega o producción! 🚀
