# 🚀 Guía de Deploy en Streamlit Cloud

## Problema Resuelto

✅ Se agregaron todas las dependencias faltantes en `requirements.txt`
✅ Se creó `packages.txt` para dependencias del sistema
✅ Se configuró `.streamlit/config.toml`

---

## 📋 Pasos para Deploy Exitoso

### 1. Actualizar repositorio local
```bash
cd /home/gjaset/Escritorio/python
git add requirements.txt packages.txt .streamlit/
git commit -m "Add deployment configuration for Streamlit Cloud"
git push origin master
```

### 2. Ir a Streamlit Cloud
- URL: https://streamlit.io/cloud
- Inicia sesión con GitHub

### 3. Crear nueva app
- Click en "New app"
- Repository: `gta-corpus-analyzer`
- Branch: `master`
- Main file: `streamlit_dashboard.py`

### 4. Esperar a que se construya
El proceso toma 2-5 minutos:
- Instala dependencias
- Compila la app
- La publica

### 5. ¡Listo! 🎉
Tu app estará disponible en:
```
https://<username>-gta-corpus-analyzer.streamlit.app
```

---

## 📦 Archivos de Configuración

### `requirements.txt`
Todas las dependencias Python necesarias:
- ✅ streamlit - Framework web
- ✅ pandas - Procesamiento de datos
- ✅ plotly - Gráficos interactivos
- ✅ networkx - Análisis de redes
- ✅ scikit-learn - ML (faltaba)
- ✅ wordcloud - Nubes de palabras
- ✅ matplotlib - Gráficos
- ✅ y más...

### `packages.txt`
Dependencias del sistema operativo:
- graphviz - Para gráficos avanzados
- libgraphviz-dev - Desarrollo graphviz

### `.streamlit/config.toml`
Configuración de Streamlit:
- Colores personalizados
- Tema oscuro
- Optimizaciones de servidor

---

## 🔧 Troubleshooting

### Si aún ves errores:

**Error: "ModuleNotFoundError"**
```bash
# Verifica que requirements.txt esté actualizado
pip freeze > requirements.txt
git add requirements.txt
git push
```

**Error: "App crashed"**
- Revisa los logs en Streamlit Cloud
- Click "Manage app" → "Settings" → "Logs"

**App lenta**
- Streamlit Cloud puede ser lento con datos grandes
- Considera cachear más datos
- Usa `@st.cache_data` agresivamente

---

## 💡 Consejos

1. **Caché agresivo**
   ```python
   @st.cache_data(ttl=3600)  # Cache 1 hora
   def load_data():
       ...
   ```

2. **Optimizar datos**
   - Comprime CSVs
   - Carga solo lo necesario

3. **Monitorear uso**
   - Streamlit Cloud limita recursos
   - Optimiza queries

---

## 📊 URL Final

Tu dashboard estará en:
```
https://<tu-username>-gta-corpus-analyzer.streamlit.app
```

Perfecto para:
- ✅ Portfolio
- ✅ LinkedIn
- ✅ Entrevistas técnicas
- ✅ Mostrar a clientes

---

## 🎯 Siguiente

Si todo funciona:
1. Comparte URL en portafolio
2. Agrega en LinkedIn
3. Menciona en CV

¡Éxito! 🚀
