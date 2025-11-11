# 🎮 GTA San Andreas - Análisis de Diálogos

Dashboard interactivo para analizar patrones lingüísticos y diálogos del videojuego **Grand Theft Auto: San Andreas**.

## 📋 Descripción

Este proyecto analiza más de 120+ personajes del juego, extrayendo y analizando:
- 📊 Palabras más frecuentes por personaje
- 🔄 Patrones de interacción entre personajes
- 📈 Métricas lingüísticas (TTR, diversidad, complejidad)
- 🌐 Red de conexiones entre personajes
- ☁️ Nubes de palabras personalizadas

## 🚀 Características

### Dashboard Interactivo
- ✅ Tema oscuro profesional
- ✅ Gráficos dinámicos con Plotly
- ✅ Comparativa de personajes
- ✅ Análisis de complejidad lingüística
- ✅ Red de interacciones visuales
- ✅ Descarga de datos en ZIP

## 📦 Requisitos

```
Python 3.8+
streamlit
pandas
plotly
networkx
scikit-learn
wordcloud
matplotlib
numpy
```

## 🔧 Instalación

### 1. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

## ▶️ Ejecución

### Iniciar el Dashboard
```bash
streamlit run streamlit_dashboard.py
```

El dashboard se abrirá en `http://localhost:8501`

### Ejecutar el Notebook
```bash
jupyter notebook GTA_San_Andreas_Dashboard.ipynb
```

## 📂 Estructura del Proyecto

```
.
├── streamlit_dashboard.py               # Dashboard principal
├── GTA_San_Andreas_Dashboard.ipynb      # Notebook de análisis
├── personajes_gta_san_andreas.csv       # Lista de personajes
├── guionGTA.txt                         # Guión original
├── requirements.txt                     # Dependencias Python
├── .gitignore                           # Configuración Git
├── .gitattributes                       # Atributos Git
├── README.md                            # Este archivo
├── resultados/
│   └── lemmatized/
│       ├── word_counts_*.csv            # Palabras por personaje (~120)
│       ├── top_words_lemmatized.csv
│       ├── top_characters_lemmatized.csv
│       ├── interaction_edges_lemmatized.csv
│       ├── interaction_matrix_lemmatized.csv
│       └── lexical_summary.csv
└── venv/                                # Entorno virtual
```

## 🎯 Uso del Dashboard

### Controles Principales

**Sidebar - 🎮 Controles:**
- 🔍 **Filtros**: Umbral mínimo de frecuencia
- 👤 **Personaje**: Selecciona personaje o "Todos"
- 📥 **Exportar**: Descarga datos en ZIP

### Secciones

#### 1️⃣ Visión General
- Métricas del personaje seleccionado
- Comparación con CJ
- Total de palabras, vocabulario, frecuencia media

#### 2️⃣ Análisis Comparativo
- Comparativa de hasta 5 personajes
- Gráfico de radar de métricas
- TTR, complejidad lingüística

#### 3️⃣ Patrones de Vocabulario
- Palabras frecuentes (gráfico de barras)
- Nube de palabras personalizable
- 5 temas de color disponibles

#### 4️⃣ Red de Interacciones
- Visualización de conexiones entre personajes
- Nodos coloreados por rol (receptivo/locuaz/equilibrado)
- Top 10 personajes por interacción

## 🎨 Tema Oscuro

Paleta de colores profesional:
- Verde: #1b5e20 (primario)
- Naranja: #ff9800 (secundario)
- Morado: #6a1b9a (terciario)

## 📊 Datos

### Personajes Analizados
- **Total**: 120+ personajes
- **Palabras**: 1 - 50,000+ por personaje
- **Vocabulario Único**: 10,000+ palabras

### Procesamiento
✅ Lematizados | ✅ Normalizados | ✅ Filtrados | ✅ Agrupados

## 📝 Cambios Recientes (v1.1)

- ✅ Tema oscuro profesional
- ✅ Red de interacciones mejorada
- ✅ Filtro de personajes actualizado (120+)
- ✅ Limpieza de archivos innecesarios
- ✅ .gitignore y .gitattributes configurados
- ✅ README completo

## 🔒 Licencia

Proyecto de análisis educativo. GTA San Andreas © Rockstar Games.

---

*Last Updated: Noviembre 2025*
