import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Upload CSV
st.sidebar.title("Upload dos Dados")
uploaded_file = st.sidebar.file_uploader("Faça upload do arquivo winequality-red.csv", type=["csv"])

# Captura o parâmetro 'grafico' da URL
query_params = st.query_params
grafico = query_params.get('grafico', 'all')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Remove a coluna 'Id' se existir
    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)

    numeric_columns = df.select_dtypes(include='number').columns

    # Visualização inicial
    if grafico in ['all', 'dataframe']:
        st.subheader("Visualização Inicial dos Dados")
        st.dataframe(df.head())

    # Distribuição da Qualidade
    if grafico in ['all', 'distribuicao']:
        st.subheader("Distribuição da Qualidade do Vinho")
        fig1 = px.histogram(df, x='quality', color_discrete_sequence=['#6E0B3C'])
        st.plotly_chart(fig1, use_container_width=True)

    # Histograma de Variáveis
    if grafico in ['all', 'histograma']:
        st.subheader("Histogramas das Variáveis Numéricas")
        selected_col = st.selectbox("Selecione uma coluna para o histograma", numeric_columns, key='hist')
        fig2 = px.histogram(df, x=selected_col, nbins=30, color_discrete_sequence=['#4B2245'])
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap com anotações numéricas
    if grafico in ['all', 'heatmap']:
        st.subheader("Mapa de Correlação entre Variáveis")
        corr_matrix = df.drop(columns=['Id'], errors='ignore').corr(numeric_only=True)

        fig3 = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index[::-1],  # Corrige a inversão vertical
                colorscale='RdPu',
                colorbar=dict(title='Correlação'),
                zmin=-1,
                zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                showscale=True
            )
        )

        fig3.update_layout(
            xaxis_title='Variáveis',
            yaxis_title='Variáveis',
            plot_bgcolor='#f8f4f9',
            paper_bgcolor='#f8f4f9'
        )

        st.plotly_chart(fig3, use_container_width=True)


    # Dispersão: Álcool x Qualidade
    if grafico in ['all', 'alcool_qualidade']:
        st.subheader("Dispersão: Álcool x Qualidade")
        fig4 = px.scatter(
            df,
            x='alcohol',
            y='quality',
            color='quality',
            opacity=0.7,
            color_continuous_scale='RdPu'
        )
        fig4.update_layout(
            plot_bgcolor='#f8f4f9',
            paper_bgcolor='#f8f4f9',
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Dispersão: Acidez x Qualidade
    if grafico in ['all', 'acidez_qualidade']:
        st.subheader("Dispersão: Acidez Volátil x Qualidade")
        fig5 = px.scatter(
            df,
            x='volatile acidity',
            y='quality',
            color='quality',
            opacity=0.7,
            color_continuous_scale='OrRd'
        )
        fig5.update_layout(
            plot_bgcolor='#f8f4f9',
            paper_bgcolor='#f8f4f9',
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Boxplot horizontal - exceto quality
    if grafico in ['all', 'boxplot']:
        st.subheader("Boxplots Horizontais das Variáveis Numéricas")

        for col in numeric_columns:
            if col != 'quality':
                fig6 = px.box(
                    df,
                    x=col,
                    points="outliers",
                    orientation='h',
                    color_discrete_sequence=['#6E0B3C']
                )
                fig6.update_layout(
                    plot_bgcolor='#f8f4f9',
                    paper_bgcolor='#f8f4f9',
                    height=400
                )
                st.plotly_chart(fig6, use_container_width=True)

    # Scatter interativo entre quaisquer duas variáveis
    if grafico in ['all', 'scatter']:
        st.subheader("Relação entre Variáveis (Scatterplot Interativo)")
        x_axis = st.selectbox("Selecione a variável para o eixo X", numeric_columns, index=0, key='scatter_x')
        y_axis = st.selectbox("Selecione a variável para o eixo Y", numeric_columns, index=1, key='scatter_y')
        fig7 = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color='quality',
            opacity=0.7,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig7, use_container_width=True)

else:
    st.warning("Por favor, faça upload do arquivo CSV para visualizar os dados.")
