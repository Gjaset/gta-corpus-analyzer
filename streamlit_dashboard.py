import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import io
import math
import zipfile
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = SCRIPT_DIR / 'resultados' / 'lemmatized'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Configuración de página
st.set_page_config(
    layout='wide',
    page_title='GTA San Andreas - Análisis de Diálogos',
    page_icon='🎮',
    initial_sidebar_state='expanded'
)

# Estilos CSS personalizados - TEMA OSCURO
st.markdown("""
<style>
    /* Fondo oscuro general */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar oscura */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 2px solid #1b5e20;
    }
    
    /* Reducir padding superior */
    .css-1v3fvcr {padding-top: 0rem !important}
    
    /* Estilo para títulos principales */
    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
        color: #f0f0f0 !important;
        text-shadow: 0 0 10px rgba(27, 94, 32, 0.3);
        font-weight: 900 !important;
    }
    
    /* Headers secundarios */
    h2 {
        font-size: 1.5rem !important;
        margin-top: 1.5rem !important;
        color: #ffffff !important;
        border-bottom: 2px solid #1b5e20 !important;
        padding-bottom: 0.5rem !important;
    }
    
    h3 {
        font-size: 1.2rem !important;
        color: #ffffff !important;
    }
    
    /* Estilo para subtítulos */
    .subtitle {
        color: #999;
        font-size: 1.1rem;
        font-style: italic;
        text-shadow: 0 0 5px rgba(255, 152, 0, 0.2);
    }
    
    /* Estilo para métricas */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #297D30 !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #999 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 1rem !important;
        color: #ff9800 !important;
    }
    
    /* Contenedor de métrica */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(27, 94, 32, 0.1) 0%, rgba(255, 152, 0, 0.05) 100%);
        border: 1px solid rgba(27, 94, 32, 0.3);
        border-radius: 8px;
        padding: 1rem !important;
    }
    
    /* Botones */
    button {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%) !important;
        color: #e0e0e0 !important;
        border: 1px solid #1b5e20 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%) !important;
        box-shadow: 0 0 15px rgba(27, 94, 32, 0.5) !important;
    }
    
    /* Selectboxes y multiselects */
    div[data-baseweb="select"] {
        background: #1a1a2e !important;
        border: 1px solid #1b5e20 !important;
        border-radius: 6px !important;
    }
    
    /* Sliders */
    div[data-baseweb="slider"] {
        color: #1b5e20 !important;
    }
    
    /* Tabs */
    div[data-baseweb="tab-list"] {
        background: #0d0d1a !important;
        border-bottom: 2px solid #1b5e20 !important;
    }
    
    button[data-baseweb="tab"] {
        color: #999 !important;
        background: transparent !important;
        font-size: 16px !important;
        padding: 20px 30px !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1b5e20 !important;
        background: linear-gradient(180deg, transparent 0%, rgba(27, 94, 32, 0.1) 100%) !important;
        border-bottom: 3px solid #1b5e20 !important;
        font-size: 30px !important;
    }
    
    /* Info boxes */
    div[data-testid="stInfo"] {
        background: rgba(27, 94, 32, 0.1) !important;
        border-left: 4px solid #1b5e20 !important;
        color: #e0e0e0 !important;
    }
    
    /* Divisores */
    hr {
        border-color: #1b5e20 !important;
        opacity: 0.3 !important;
    }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] {
        background: #1a1a2e !important;
    }
    
    /* Caption y text */
    .stCaption {
        color: #888 !important;
    }
</style>
""", unsafe_allow_html=True)

# Paleta de colores
PA_PRIMARY = '#1b5e20'   # verde principal
PA_SECOND = '#ff9800'    # naranja secundario
PA_TERTIARY = '#6a1b9a'  # morado terciario
HEAT_SCALE = ['#e8f5e9', PA_PRIMARY, '#66bb6a', PA_SECOND, PA_TERTIARY]

# Cargar datos
@st.cache_data
def load_data():
    data = {}
    
    # Verificar que la carpeta existe
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Carpeta de resultados no encontrada: {RESULTS_DIR}")
    
    try:
        data['top_words'] = pd.read_csv(RESULTS_DIR / 'top_words_lemmatized.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No se encontró top_words_lemmatized.csv en {RESULTS_DIR}: {e}")
    
    try:
        data['top_characters'] = pd.read_csv(RESULTS_DIR / 'top_characters_lemmatized.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No se encontró top_characters_lemmatized.csv en {RESULTS_DIR}: {e}")
    
    try:
        data['interaction_matrix'] = pd.read_csv(RESULTS_DIR / 'interaction_matrix_lemmatized.csv', index_col=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No se encontró interaction_matrix_lemmatized.csv en {RESULTS_DIR}: {e}")
    
    try:
        data['interaction_edges'] = pd.read_csv(RESULTS_DIR / 'interaction_edges_lemmatized.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No se encontró interaction_edges_lemmatized.csv en {RESULTS_DIR}: {e}")
    
    per_char = {}
    for p in RESULTS_DIR.glob('word_counts_*.csv'):
        try:
            name = p.stem.replace('word_counts_', '')
            df = pd.read_csv(p)
            if 'lemma' in df.columns and 'count' in df.columns:
                per_char[name] = df
        except Exception as e:
            st.warning(f"Error cargando {p.name}: {e}")
            continue
    
    if not per_char:
        st.error("No se pudieron cargar los datos de personajes")
        st.stop()
    
    data['per_char'] = per_char
    return data

try:
    data = load_data()
except Exception as e:
    st.error(f'Error cargando datos: {str(e)}')
    st.error(f'Ruta esperada: {RESULTS_DIR}')
    st.error(f'¿Existe la carpeta?: {RESULTS_DIR.exists()}')
    st.stop()

# Funciones auxiliares
def calculate_metrics(df):
    total_words = df['count'].sum()
    unique_words = len(df)
    avg_usage = total_words / unique_words if unique_words > 0 else 0
    return total_words, unique_words, avg_usage

@st.cache_data
def load_characters():
    # Usar directamente los personajes disponibles en data['per_char']
    personajes = sorted(list(data['per_char'].keys()))
    
    # Coloca CJ (Carl Johnson) primero, luego los demás
    if 'CJ (Carl Johnson)' in personajes:
        personajes.remove('CJ (Carl Johnson)')
        personajes = ['Todos los personajes', 'CJ (Carl Johnson)'] + personajes
    else:
        personajes = ['Todos los personajes'] + personajes
    
    return personajes

# Título principal y descripción
st.title('El lenguaje de Los Santos: Un viaje por las voces de GTA San Andreas')
st.markdown(
    '<p class="subtitle">Análisis detallado de los diálogos y patrones lingüísticos del juego</p>',
    unsafe_allow_html=True
)

# Sidebar con controles
with st.sidebar:
    st.header('🎮 Controles')
    
    # Filtros generales
    st.subheader('🔍 Filtros')
    # Sin umbral mínimo de frecuencia - mostrar todas las palabras
    min_word_count = 1
    
    # Filtro de rango de frecuencia para palabras
    col1, col2 = st.columns(2)
    with col1:
        freq_min = st.number_input('Freq. mín.', min_value=1, max_value=100, value=1, step=1)
    with col2:
        freq_max = st.number_input('Freq. máx.', min_value=1, max_value=500, value=200, step=10)
    
    # Filtro de longitud de palabras
    word_length_filter = st.slider(
        'Longitud mínima de palabra (caracteres)',
        min_value=1,
        max_value=20,
        value=1,
        help='Mostrar solo palabras con esta longitud mínima'
    )
    
    # Tipo de visualización en Patrones de Vocabulario
    viz_type = st.radio(
        'Vista de palabras frecuentes',
        ['Tabla', 'Gráfico de barras', 'Ambas'],
        horizontal=True,
        help='Selecciona cómo ver las palabras más frecuentes'
    )
    
    st.markdown('---')
    
    # Selección de personaje
    st.subheader('👤 Personaje')
    character = st.selectbox(
        'Seleccionar personaje para análisis',
        load_characters(),
        index=1,
        help='Elige un personaje específico o "Todos los personajes" para una vista general'
    )
    
    st.markdown('---')
    
    # Exportar datos
    st.subheader('📥 Exportar')
    if st.button('Descargar datos completos'):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zip_file.writestr(f'{name}.csv', csv_buffer.getvalue())
        
        st.download_button(
            '📥 Descargar ZIP con datos',
            data=zip_buffer.getvalue(),
            file_name='gta_sa_dialogue_data.zip',
            mime='application/zip'
        )

# 1. VISIÓN GENERAL
st.header('1️⃣ Visión General', help='Resumen general de los diálogos')

# Métricas principales
if character == 'Todos los personajes':
    total_words = sum(df['count'].sum() for df in data['per_char'].values())
    unique_words = len(data['top_words'])
    avg_usage = total_words / unique_words if unique_words > 0 else 0
    delta = None
else:
    if character in data['per_char']:
        char_data = data['per_char'][character]
        total_words, unique_words, avg_usage = calculate_metrics(char_data)
        
        # Comparar con CJ
        cj_key = next((k for k in data['per_char'].keys() if 'CJ' in k), None)
        if cj_key and character != cj_key:
            cj_data = data['per_char'][cj_key]
            cj_total, _, _ = calculate_metrics(cj_data)
            delta = f"{((total_words - cj_total) / cj_total * 100):.1f}%"
        else:
            delta = None
    else:
        total_words = unique_words = avg_usage = 0
        delta = None

# Mostrar métricas principales
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "📊 Total de Palabras",
        f"{total_words:,}",
        delta,
        help="Número total de palabras utilizadas"
    )

with col2:
    st.metric(
        "🔤 Vocabulario Único",
        f"{unique_words:,}",
        help="Cantidad de palabras diferentes"
    )

with col3:
    st.metric(
        "📈 Frecuencia Media",
        f"{avg_usage:.1f}",
        help="Promedio de uso por palabra"
    )

# 2. ANÁLISIS COMPARATIVO
st.header('2️⃣ Análisis Comparativo', help='Comparación entre personajes')

tab1, tab2 = st.tabs(["📊 Comparativa General", "🔄 Análisis Detallado"])

with tab1:
    # Selector de personajes y métricas
    col_comp1, col_comp2 = st.columns([2, 2])
    with col_comp1:
        personajes_seleccionados = st.multiselect(
            'Selecciona personajes para comparar',
            options=[p for p in data['per_char'].keys() if p != 'Todos los personajes'],
            default=['CJ (Carl Johnson)', next((p for p in data['per_char'].keys() if p != 'CJ (Carl Johnson)'), None)],
            max_selections=5
        )

    with col_comp2:
        metrica = st.selectbox(
            'Métrica a comparar',
            ['Palabras Totales', 'Palabras Únicas', 'Longitud Promedio', 'TTR', 'Frecuencia de Palabras Comunes']
        )
    
    if personajes_seleccionados:
        # Recopilar datos para la comparación
        datos_comparacion = []
        for personaje in personajes_seleccionados:
            if personaje in data['per_char']:
                char_data = data['per_char'][personaje]
                total_words = char_data['count'].sum()
                unique_words = len(char_data)
                avg_word_length = char_data.apply(lambda x: len(str(x['lemma'])) * x['count'], axis=1).sum() / total_words
                ttr = unique_words / total_words if total_words > 0 else 0
                
                # Calcular frecuencia de palabras comunes (top 10 globales)
                common_words = set(data['top_words'].head(10)['lemma'])
                common_words_freq = sum(char_data[char_data['lemma'].isin(common_words)]['count']) / total_words
                
                datos_comparacion.append({
                    'Personaje': personaje,
                    'Palabras Totales': total_words,
                    'Palabras Únicas': unique_words,
                    'Longitud Promedio': avg_word_length,
                    'TTR': ttr,
                    'Frecuencia de Palabras Comunes': common_words_freq
                })
        
        df_comparacion = pd.DataFrame(datos_comparacion)
        
        # Crear visualización de pastel
        total = df_comparacion[metrica].sum()
        
        if metrica in ['TTR', 'Longitud Promedio', 'Frecuencia de Palabras Comunes']:
            hover_text = [f'{p}<br>{m:.3f}' for p, m in zip(df_comparacion['Personaje'], df_comparacion[metrica])]
            valores = df_comparacion[metrica]
        else:
            porcentajes = (df_comparacion[metrica] / total * 100)
            hover_text = [f'{p}<br>{v:,.0f}<br>({pct:.1f}%)' 
                         for p, v, pct in zip(df_comparacion['Personaje'], 
                                            df_comparacion[metrica], 
                                            porcentajes)]
            valores = porcentajes

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Pie(
            labels=df_comparacion['Personaje'],
            values=valores,
            hovertext=hover_text,
            hoverinfo='text',
            textinfo='label+percent',
            hole=0.4,
            marker=dict(
                colors=[PA_PRIMARY if 'CJ' in p else PA_SECOND for p in df_comparacion['Personaje']]
            ),
            rotation=90
        ))
        
        fig_comp.update_layout(
            title=dict(text=f'Distribución de {metrica}', font=dict(color='#1b5e20', size=16)),
            annotations=[dict(
                text=metrica,
                x=0.5,
                y=0.5,
                font_size=12,
                font_color='#e0e0e0',
                showarrow=False
            )],
            showlegend=True,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(color='#e0e0e0')
            ),
            paper_bgcolor='#0f0f1e',
            plot_bgcolor='rgba(15, 15, 30, 0.5)',
            font=dict(color='#e0e0e0')
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)

with tab2:
    st.subheader("Análisis de Complejidad Lingüística")
    
    # Encontrar CJ
    cj_key = next((k for k in data['per_char'].keys() if 'CJ' in k), None)
    
    if cj_key:
        # Controles para seleccionar personajes a comparar
        st.markdown("**Selecciona personajes para analizar**")
        
        available_chars = sorted([c for c in data['per_char'].keys()])
        default_selected = [cj_key]  # CJ por defecto
        
        # Si hay un personaje seleccionado actualmente, incluirlo también
        if character != 'Todos los personajes' and character in available_chars and character != cj_key:
            default_selected.append(character)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_chars = st.multiselect(
                'Personajes a analizar:',
                options=available_chars,
                default=default_selected,
                key='complexity_chars'
            )
        
        with col2:
            max_comparisons = st.number_input('Máx. personajes:', min_value=1, max_value=8, value=5)
            if len(selected_chars) > max_comparisons:
                selected_chars = selected_chars[:max_comparisons]
                st.warning(f'Limitado a {max_comparisons} personajes')
        
        with col3:
            show_table = st.checkbox('Mostrar tabla', value=True)
        
        if selected_chars:
            # Función auxiliar para calcular métricas
            def calculate_metrics(char_name):
                char_data = data['per_char'][char_name]
                total_words = char_data['count'].sum()
                unique_words = len(char_data)
                avg_word_length = char_data.apply(lambda x: len(str(x['lemma'])) * x['count'], axis=1).sum() / total_words
                
                ttr = unique_words / total_words if total_words > 0 else 0
                
                return {
                    'name': char_name,
                    'total_words': total_words,
                    'unique_words': unique_words,
                    'avg_length': avg_word_length,
                    'ttr': ttr
                }
            
            # Calcular métricas para normalización
            all_metrics = [calculate_metrics(char) for char in selected_chars]
            max_avg_length = max(m['avg_length'] for m in all_metrics)
            
            # Construir datos para el gráfico
            metrics_data = []
            for metrics in all_metrics:
                normalized_length = metrics['avg_length'] / max_avg_length if max_avg_length > 0 else 0
                complexity = (metrics['ttr'] + normalized_length) / 2
                
                metrics_data.append({
                    'name': metrics['name'],
                    'Diversidad Léxica': min(metrics['ttr'], 1.0),
                    'Longitud Palabras': normalized_length,
                    'Complejidad General': complexity,
                    'Densidad Vocab.': min(metrics['unique_words'] / max(50, metrics['total_words'] / 5), 1.0)
                })
            
            # Colores para cada personaje
            colors = ['#4ECDC4', '#FF6B6B', '#95E1D3', '#F38181', '#AA96DA', '#FCBAD3', '#FFD93D', '#6BCB77']
            
            categories = ['Diversidad Léxica', 'Longitud Palabras', 'Complejidad General', 'Densidad Vocab.']
            fig_radar = go.Figure()
            
            for idx, metrics in enumerate(metrics_data):
                is_cj = metrics['name'] == cj_key
                line_width = 3 if is_cj else 2
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=[metrics[cat] for cat in categories],
                    theta=categories,
                    fill='toself',
                    name=metrics['name'],
                    line=dict(color=colors[idx % len(colors)], width=line_width),
                    fillcolor=f"rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, {'0.4' if is_cj else '0.2'})",
                    opacity=1.0
                ))
            
            title_text = 'Análisis de Complejidad Lingüística'
            if cj_key in selected_chars:
                title_text += ' (CJ resaltado en línea más gruesa)'
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor='rgba(27, 94, 32, 0.4)',
                        tickcolor='#1b5e20',
                        tickfont=dict(color='#e0e0e0', size=10)
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='#e0e0e0', size=11),
                        gridcolor='rgba(27, 94, 32, 0.2)'
                    ),
                    bgcolor='rgba(15, 15, 30, 0.3)'
                ),
                showlegend=True,
                title=dict(
                    text=title_text,
                    font=dict(color='#ffffff', size=16),
                    x=0.5,
                    xanchor='center'
                ),
                paper_bgcolor='#0f0f1e',
                font=dict(color='#e0e0e0', size=12),
                legend=dict(
                    font=dict(color='#e0e0e0', size=10),
                    x=1.05,
                    y=1,
                    bgcolor='rgba(0, 0, 0, 0)',
                    bordercolor='rgba(27, 94, 32, 0.3)',
                    borderwidth=1
                ),
                height=500,
                margin=dict(l=80, r=150, t=100, b=80)
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Tabla comparativa
            if show_table:
                st.subheader("📊 Comparación Detallada")
                comparison_df = pd.DataFrame([
                    {
                        'Personaje': m['name'],
                        'Diversidad Léxica': f"{m['Diversidad Léxica']:.3f}",
                        'Longitud Palabras': f"{m['Longitud Palabras']:.3f}",
                        'Complejidad': f"{m['Complejidad General']:.3f}",
                        'Densidad': f"{m['Densidad Vocab.']:.3f}"
                    }
                    for m in metrics_data
                ])
                st.dataframe(comparison_df, use_container_width=True)
            
            # Mostrar explicación de métricas
            with st.expander("📊 Explicación de métricas", expanded=False):
                st.markdown("""
                **Diversidad Léxica (TTR)**: Ratio de palabras únicas vs total de palabras. 
                - 0.2-0.3: Vocabulario repetitivo
                - 0.3-0.5: Vocabulario variado (ideal)
                - 0.5+: Muy diverso pero menos frecuencia
                
                **Longitud de Palabras**: Promedio de caracteres por palabra normalizado.
                - Valores altos indican palabras más largas y complejas
                
                **Complejidad General**: Combinación de diversidad y longitud de palabras.
                - Métrica agregada de complejidad lingüística
                
                **Densidad de Vocabulario**: Proporción de palabras únicas por cada 5 palabras.
                - Indica cuán "fresco" es el vocabulario usado
                """)


# 3. PATRONES DE VOCABULARIO
st.header('3️⃣ Patrones de Vocabulario', help='Análisis del uso de palabras')

tab3, tab4 = st.tabs(["📝 Palabras Frecuentes", "☁️ Nube de Palabras"])

with tab3:
    col1, col2 = st.columns([3, 1])
    with col1:
        top_n = st.slider('Número de palabras a mostrar', 5, 100, 30)
        sort_dir = st.selectbox('Ordenar por', ['Frecuencia ↓', 'Frecuencia ↑'])
    
    if character == 'Todos los personajes':
        df_display = data['top_words'].copy()
        title = 'Palabras más frecuentes en todo el juego'
    else:
        if character in data['per_char']:
            df_display = data['per_char'][character].copy()
            df_display.columns = ['lemma', 'count']
            title = f'Palabras más frecuentes de {character}'
        else:
            df_display = pd.DataFrame(columns=['lemma', 'count'])
            title = 'No hay datos disponibles'
    
    # Filtrar y ordenar
    df_display = df_display[(df_display['count'] >= freq_min) & (df_display['count'] <= freq_max)]
    df_display = df_display[df_display['lemma'].str.len() >= word_length_filter]
    df_display = df_display.sort_values('count', ascending=sort_dir == 'Frecuencia ↑')
    df_display = df_display.head(top_n)
    
    if not df_display.empty:
        # Mostrar según tipo de visualización seleccionado
        show_table = viz_type in ['Tabla', 'Ambas']
        show_chart = viz_type in ['Gráfico de barras', 'Ambas']
        
        if show_chart:
            fig = px.bar(df_display, 
                        x='count', 
                        y='lemma',
                        orientation='h',
                        title=title,
                        labels={'count': 'Frecuencia', 'lemma': 'Palabra'},
                        color_discrete_sequence=[PA_PRIMARY])
            
            fig.update_layout(
                showlegend=False,
                title_x=0.5,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title_font=dict(size=12), gridcolor='rgba(27, 94, 32, 0.2)'),
                yaxis=dict(title_font=dict(size=12)),
                plot_bgcolor='rgba(15, 15, 30, 0.5)',
                paper_bgcolor='#0f0f1e',
                font=dict(color='#e0e0e0'),
                title_font=dict(color='#1b5e20', size=16)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if show_table:
            st.subheader('📋 Datos Detallados')
            st.dataframe(
                df_display.reset_index(drop=True),
                use_container_width=True,
                height=400
            )
    else:
        st.info('No hay palabras que cumplan con los filtros actuales')

with tab4:
    col_wc1, col_wc2 = st.columns([3, 1])
    with col_wc2:
        wc_height = st.slider('Tamaño', 300, 800, 500, step=50)
        wc_width = int(wc_height * 1.6)
        color_theme = st.selectbox(
            'Tema de color',
            ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        )
    
    with col_wc1:
        if character == 'Todos los personajes':
            words_freq = data['top_words'].set_index('lemma')['count'].to_dict()
            title = "Nube de palabras del juego completo"
        else:
            if character in data['per_char']:
                char_data = data['per_char'][character]
                words_freq = char_data.set_index('lemma')['count'].to_dict()
                title = f"Nube de palabras de {character}"
            else:
                words_freq = {}
                title = "No hay datos disponibles"

        if words_freq:
            # Filtrar por frecuencia mínima
            words_freq = {word: freq for word, freq in words_freq.items() if freq >= min_word_count}
            
            if words_freq:
                wc = WordCloud(
                    width=wc_width,
                    height=wc_height,
                    background_color='#0f0f1e',
                    colormap=color_theme,
                    max_words=100,
                    prefer_horizontal=0.7,
                    relative_scaling=0.5,
                    min_font_size=8,
                    max_font_size=80,
                    random_state=42
                )
                wc.generate_from_frequencies(words_freq)
                
                fig = plt.figure(figsize=(12, 7.5), facecolor='#0f0f1e', edgecolor='#1b5e20')
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(title, pad=20, size=14, color='#1b5e20', fontweight='bold')
                plt.tight_layout(pad=0)
                
                st.pyplot(fig)
            else:
                st.info('No hay palabras que cumplan con el umbral mínimo de frecuencia')

# 4. RED DE INTERACCIONES
st.header('4️⃣ Red de Interacciones', help='Visualización de conexiones entre personajes')

# Control de umbral
min_edge = st.slider(
    'Umbral mínimo de interacciones',
    1, 50, 3,
    help='Muestra conexiones con al menos este número de interacciones'
)

edges = data['interaction_edges']
edges_filtered = edges[edges['count'] >= min_edge]

if edges_filtered.empty:
    st.info('No hay interacciones que superen el umbral seleccionado')
else:
    # Crear grafo dirigido
    G = nx.DiGraph()
    for _, r in edges_filtered.iterrows():
        G.add_edge(r['source'], r['target'], weight=r['count'])

    # Calcular métricas de importancia
    degrees = dict(G.degree(weight='weight'))
    in_degrees = dict(G.in_degree(weight='weight'))
    out_degrees = dict(G.out_degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    
    # Tamaño de nodos basado en grado (mayor interacción = nodo más grande)
    sizes = {n: 15 + 35 * (degrees.get(n,0) / max_deg) for n in G.nodes()}
    
    # Layout mejorado con más iteraciones
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Crear trazos de aristas CON INFORMACIÓN VISUAL
    edge_traces = []
    edge_weights = []
    
    for u, v, data_e in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data_e['weight']
        edge_weights.append(weight)
        
        # Ancho de línea proporcional al peso
        width = 1 + (weight / max(G[u][v]['weight'] for u, v in G.edges())) * 5
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=width,
                color=f'rgba(187, 187, 187, {0.3 + 0.7 * (weight / max(edge_weights))})'
            ),
            hovertext=f'{u} → {v}<br>Interacciones: {weight}',
            hoverinfo='text',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Crear nodos con colores dinámicos y efectos
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_customdata = []
    
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        
        total_inter = degrees.get(n, 0)
        in_inter = in_degrees.get(n, 0)
        out_inter = out_degrees.get(n, 0)
        
        hover_info = (
            f'<b>{n}</b><br>'
            f'Interacciones totales: {total_inter}<br>'
            f'↓ Recibidas: {in_inter}<br>'
            f'↑ Enviadas: {out_inter}'
        )
        node_text.append(hover_info)
        node_size.append(sizes.get(n, 15))
        
        # Color basado en si es más locuaz (out) o receptivo (in)
        if in_inter > out_inter:
            node_color.append(PA_PRIMARY)  # Verde - receptivo
        elif out_inter > in_inter:
            node_color.append(PA_SECOND)  # Naranja - locuaz
        else:
            node_color.append(PA_TERTIARY)  # Morado - equilibrado
        
        node_customdata.append(f'{n}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertext=node_text,
        hoverinfo='text',
        text=[n.split('(')[0].strip()[:15] for n in G.nodes()],  # Etiquetas abreviadas
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial Black'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='rgba(50,50,50,0.8)'),
            opacity=0.9
        ),
        showlegend=False
    )

    # Crear figura con todas las aristas y nodos
    fig_net = go.Figure(data=edge_traces + [node_trace])
    
    fig_net.update_layout(
        title={
            'text': '🔗 Red de Interacciones entre Personajes',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1b5e20'}
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10, l=10, r=10, t=40),
        height=700,
        plot_bgcolor='rgba(15, 15, 30, 0.8)',
        paper_bgcolor='#0f0f1e',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(color='#e0e0e0', family='Arial')
    )
    
    # Agregar anotaciones legendarias
    fig_net.add_annotation(
        text=(
            f'<b>Leyenda:</b><br>'
            f'🟢 Verde: Más receptivo (recibe más diálogos)<br>'
            f'🟠 Naranja: Más locuaz (emite más diálogos)<br>'
            f'🟣 Morado: Equilibrado<br>'
            f'<i>Tamaño: Importancia en la red | Grosor líneas: Frecuencia de interacción</i>'
        ),
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor='rgba(15, 15, 30, 0.95)',
        bordercolor='#1b5e20',
        borderwidth=2,
        borderpad=10,
        font=dict(size=10, color='#e0e0e0'),
        align='left'
    )
    
    st.plotly_chart(fig_net, use_container_width=True, config=dict(
        scrollZoom=True,
        displayModeBar=True,
        responsive=True
    ))
    
    # Agregar tabla de resumen debajo
    st.subheader('📊 Estadísticas de la Red')
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric('Total de Personajes', len(G.nodes()))
    
    with col_stats2:
        st.metric('Total de Interacciones', len(G.edges()))
    
    with col_stats3:
        avg_connections = np.mean(list(degrees.values())) if degrees else 0
        st.metric('Conexiones Promedio', f'{avg_connections:.1f}')
    
    # Tabla de top personajes por interacción
    st.subheader('🎭 Top Personajes por Interacción')
    
    top_personas = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    df_top = pd.DataFrame([
        {
            'Personaje': name,
            'Interacciones': count,
            'Tipo': '↓ Receptivo' if in_degrees.get(name, 0) > out_degrees.get(name, 0) else ('↑ Locuaz' if out_degrees.get(name, 0) > in_degrees.get(name, 0) else '⚖️ Equilibrado')
        }
        for name, count in top_personas
    ])
    
    st.dataframe(df_top, use_container_width=True, hide_index=True)

# Pie de página
st.markdown('---')
st.caption("""
💡 **Guía del dashboard:**
- Usa el selector de personaje en la barra lateral para análisis específicos
- Ajusta los filtros para personalizar las visualizaciones
- Descarga los datos usando los botones de exportación
""")
st.caption('Dashboard generado a partir de datos lematizados de GTA San Andreas')