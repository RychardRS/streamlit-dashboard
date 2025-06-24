import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Análise de Vinhos",
    layout="wide",                    
    initial_sidebar_state="collapsed"
)

st.title("Análise Exploratória - Qualidade de Vinhos 🍷")
st.markdown("""
### Disciplina: Ciência de Dados  
**Grupo:**  
- Rychardson Ribeiro de Souza  
- Pedro Henrique Leite Santos

---  
""")

default_csv_path = os.path.join(os.path.dirname(__file__), "WineQT.csv")

st.sidebar.title("Upload dos Dados")
uploaded_file = st.sidebar.file_uploader(
    "Faça upload do arquivo winequality-red.csv", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Nenhum arquivo enviado — usando o CSV padrão.")
    df = pd.read_csv(default_csv_path)

# Remove coluna “Id”, caso exista
df = df.drop(columns=["Id"], errors="ignore")
numeric_cols = df.select_dtypes("number").columns

def plot_centered(fig):
    st.markdown(
        "<div style='max-width:75%;margin:0 auto;'>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def dataframe_centered(data):
    st.markdown(
        "<div style='max-width:75%;margin:0 auto;'>",
        unsafe_allow_html=True,
    )
    st.dataframe(data, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Captura o parâmetro 'grafico' da URL
grafico = st.query_params.get("grafico", "all")

# Visualização inicial do dataframe
if grafico in ("all", "dataframe"):
    st.subheader("Visualização Inicial dos Dados")
    dataframe_centered(df.head())

# Distribuição da Qualidade
if grafico in ("all", "distribuicao"):
    st.subheader("Distribuição da Qualidade do Vinho")
    fig = px.histogram(df, x="quality", color_discrete_sequence=["#6E0B3C"])
    plot_centered(fig)

# Histograma de Variáveis
if grafico in ("all", "histograma"):
    st.subheader("Histogramas das Variáveis Numéricas")
    col = st.selectbox("Selecione a variável", numeric_cols, key="hist")
    fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#4B2245"])
    plot_centered(fig)

# Heatmap de Correlação com valores
if grafico in ("all", "heatmap"):
    st.subheader("Mapa de Correlação entre Variáveis")
    corr = df.corr(numeric_only=True)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdPu",
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="Correlação"),
        )
    )
    fig.update_layout(
        xaxis_title="Variáveis",
        yaxis_title="Variáveis",
        plot_bgcolor="#f8f4f9",
        paper_bgcolor="#f8f4f9",
    )
    plot_centered(fig)

# Dispersão Álcool x Qualidade
if grafico in ("all", "alcool_qualidade"):
    st.subheader("Dispersão: Álcool × Qualidade")
    fig = px.scatter(
        df,
        x="alcohol",
        y="quality",
        color="quality",
        opacity=0.75,
        color_continuous_scale="RdPu",
    )
    plot_centered(fig)

# Dispersão Acidez x Qualidade
if grafico in ("all", "acidez_qualidade"):
    st.subheader("Dispersão: Acidez Volátil × Qualidade")
    fig = px.scatter(
        df,
        x="volatile acidity",
        y="quality",
        color="quality",
        opacity=0.75,
        color_continuous_scale="OrRd",
    )
    plot_centered(fig)

# Boxplots horizontais
if grafico in ("all", "boxplot"):
    st.subheader("Boxplots Horizontais das Variáveis Numéricas")
    for col in numeric_cols:
        if col != "quality":
            fig = px.box(
                df,
                x=col,
                orientation="h",
                points="outliers",
                color_discrete_sequence=["#6E0B3C"],
            )
            fig.update_layout(height=400)
            plot_centered(fig)

# Scatter entre quaisquer duas variáveis
if grafico in ("all", "scatter"):
    st.subheader("Relação entre Variáveis (Scatterplot Interativo)")
    eixo_x = st.selectbox("Eixo X", numeric_cols, key="scatter_x")
    eixo_y = st.selectbox("Eixo Y", numeric_cols, key="scatter_y")
    fig = px.scatter(
        df,
        x=eixo_x,
        y=eixo_y,
        color="quality",
        opacity=0.7,
        color_continuous_scale="Viridis",
    )
    plot_centered(fig)
