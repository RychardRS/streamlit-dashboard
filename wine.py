import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# --- Bibliotecas de Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuração da Página ---
st.set_page_config(layout="wide")

# --- Funções de Plotagem ---
def plot_classification_report(report_dict, title):
    """Gera uma tabela formatada para o relatório de classificação."""
    st.write(f"**{title}:**")
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)
    
    with st.expander("O que significam estas métricas?"):
        st.info("""
        **Accuracy (Acurácia):** A percentagem de previsões corretas em relação ao total. É a métrica mais simples, mas pode ser enganadora em datasets desbalanceados.
        """)
        st.info("""
        **Precision (Precisão):** De tudo que o modelo classificou como uma classe, quantos ele acertou? 
        *Ex: De todos os vinhos que o modelo previu como 'Excelente', qual a porcentagem que realmente era 'Excelente'?*
        """)
        st.info("""
        **Recall (Revocação):** De todos os exemplos reais de uma classe, quantos o modelo conseguiu encontrar?
        *Ex: De todos os vinhos que realmente são 'Excelentes' no dataset, qual a porcentagem que o modelo identificou corretamente?*
        """)
        st.info("""
        **F1-Score:** Uma média harmônica entre Precisão e Recall. É uma ótima métrica geral, especialmente quando as classes são desbalanceadas.
        """)
        st.info("""
        **Support (Suporte):** O número de ocorrências reais de cada classe no conjunto de dados. Ajuda a dar contexto às outras métricas.
        """)
        st.info("""
        **Macro Avg (Média Macro):** A média simples das métricas (ex: F1-Score) para cada classe. Trata todas as classes com o mesmo peso, independentemente do seu tamanho.
        """)
        st.info("""
        **Weighted Avg (Média Ponderada):** A média das métricas ponderada pelo suporte de cada classe. Dá mais importância às classes com mais amostras.
        """)

def plot_feature_importance(importance_df, title):
    """Gera o gráfico de barras para a importância das características."""
    st.write(f"**{title}:**")
    st.write("Mostra quais características físico-químicas mais influenciaram o modelo para tomar suas decisões.")
    fig = px.bar(importance_df, 
                 x='importance', 
                 y='feature',
                 orientation='h',
                 color='importance', 
                 color_continuous_scale='RdPu')
    fig.update_layout(yaxis_title='Característica', xaxis_title='Importância')
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(y_true, y_pred, title, labels):
    """Gera o gráfico da matriz de confusão."""
    st.write(f"**{title}:**")
    st.write("Mostra os acertos e erros do modelo. A diagonal principal representa as classificações corretas.")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    labels_for_plot = [str(l) for l in labels]
    fig = px.imshow(cm, text_auto=True,
                    labels=dict(x="Qualidade Prevista", y="Qualidade Real"),
                    x=labels_for_plot,
                    y=labels_for_plot,
                    color_continuous_scale='RdPu')
    st.plotly_chart(fig, use_container_width=True)

# --- Funções de Cache para os Modelos ---
@st.cache_data
def train_model_1(df):
    """Treina e avalia o modelo de classificação original."""
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Para o relatório de classificação, usamos todas as labels possíveis para ter uma visão completa.
    all_possible_labels = sorted(y.unique().tolist())
    report_dict = classification_report(y_test, y_pred, labels=all_possible_labels, zero_division=0, output_dict=True)

    # --- AJUSTE APLICADO AQUI ---
    # Para a matriz de confusão, é crucial usar apenas as labels que realmente existem
    # nos dados de teste (y_test) ou nas previsões (y_pred) para evitar o ValueError.
    # Combinamos os dois arrays, pegamos os valores únicos e os ordenamos.
    labels_for_confusion_matrix = sorted(np.unique(np.concatenate((y_test, y_pred))))
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # Retornamos as labels corretas para a matriz de confusão.
    return accuracy, report_dict, feature_importance, y_test, y_pred, labels_for_confusion_matrix

@st.cache_data
def train_model_2(df):
    """Treina e avalia o modelo de classificação otimizado."""
    df_cat = df.copy()
    def categorize_quality(quality):
        if quality <= 4: return 'Ruim'
        elif quality <= 6: return 'Médio'
        else: return 'Excelente'
    df_cat['categoria_qualidade'] = df_cat['quality'].apply(categorize_quality)

    X_cat = df_cat.drop(['quality', 'categoria_qualidade'], axis=1)
    y_cat = df_cat['categoria_qualidade']
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

    model_cat = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model_cat.fit(X_train_cat, y_train_cat)
    y_pred_cat = model_cat.predict(X_test_cat)
    
    accuracy_cat = accuracy_score(y_test_cat, y_pred_cat)
    report_cat_dict = classification_report(y_test_cat, y_pred_cat, labels=['Ruim', 'Médio', 'Excelente'], output_dict=True)
    feature_importance_cat = pd.DataFrame({
        'feature': X_cat.columns,
        'importance': model_cat.feature_importances_
    }).sort_values('importance', ascending=True)
    
    return accuracy_cat, report_cat_dict, feature_importance_cat, y_test_cat, y_pred_cat

# --- Carregamento dos Dados ---
try:
    # Tenta carregar o arquivo do mesmo diretório do script.
    # Isso torna o script mais portável.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_path = os.path.join(base_dir, 'WineQT.csv')
    df = pd.read_csv(default_csv_path)
except (FileNotFoundError, NameError):
    # Fallback para o caso de estar rodando em um ambiente onde __file__ não é definido (ex: alguns notebooks)
    try:
        df = pd.read_csv('WineQT.csv')
    except FileNotFoundError:
        st.error("Arquivo 'WineQT.csv' não encontrado. Por favor, coloque o arquivo na mesma pasta do script.")
        st.stop()


if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)

# --- Sidebar ---
st.sidebar.title("Análise de Vinhos")
page = st.sidebar.radio("", 
                        ["Análise Exploratória", 
                         "Análise Preditiva", 
                         "Análise Preditiva Otimizada", 
                         "Conclusões e Comparativo"])

# --- Corpo Principal ---
if page == "Análise Exploratória":
    st.title("Análise Exploratória")
    st.subheader("Visualização Inicial dos Dados")
    st.dataframe(df.head())
    st.subheader("Distribuição da Qualidade do Vinho")
    fig1 = px.histogram(df, x='quality', color_discrete_sequence=['#6E0B3C'])
    st.plotly_chart(fig1, use_container_width=True)
    st.subheader("Mapa de Correlação entre Variáveis")
    corr_matrix = df.corr(numeric_only=True)
    fig3 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
        colorscale='RdPu', zmin=-1, zmax=1, text=corr_matrix.round(2).values,
        texttemplate="%{text}", showscale=True))
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Análise Preditiva":
    st.title("Análise Preditiva: Prevendo a Nota Original")
    st.info("""
    **Método Utilizado: Random Forest**

    Escolhemos o algoritmo Random Forest por sua alta precisão e robustez. Ele funciona criando múltiplas árvores de decisão e combinando seus resultados, o que o torna menos propenso a superajuste (overfitting). Além disso, ele nos permite calcular a 'importância' de cada característica, mostrando quais fatores mais influenciam a qualidade do vinho.
    """)
    st.info("Nesta análise, treinamos um modelo para prever a nota exata de qualidade (3 a 8).")
    
    st.write("**Distribuição das Classes Originais:**")
    st.write("O gráfico abaixo mostra o desbalanceamento das notas, com a maioria dos vinhos concentrados nas qualidades 5 e 6.")
    fig_dist = px.histogram(df, x='quality', color_discrete_sequence=['#6E0B3C'])
    st.plotly_chart(fig_dist, use_container_width=True)
    
    accuracy, report_dict, feature_importance, y_test, y_pred, classes = train_model_1(df)
    st.metric(label="Acurácia no Teste", value=f"{accuracy:.2%}")
    plot_classification_report(report_dict, "Relatório de Classificação")
    plot_confusion_matrix(y_test, y_pred, "Matriz de Confusão", classes)
    plot_feature_importance(feature_importance, "Importância das Características")

elif page == "Análise Preditiva Otimizada":
    st.title("Análise Preditiva Otimizada: Prevendo Categorias")
    st.info("""
    **Método Utilizado: Random Forest**

    Escolhemos o algoritmo Random Forest por sua alta precisão e robustez. Ele funciona criando múltiplas árvores de decisão e combinando seus resultados, o que o torna menos propenso a superajuste (overfitting). Além disso, ele nos permite calcular a 'importância' de cada característica, mostrando quais fatores mais influenciam a qualidade do vinho.
    """)
    st.info("Para melhorar a performance, agrupamos as notas em 3 categorias, criando um problema mais simples e balanceado para o modelo.")
    st.markdown("""
    As categorias foram definidas da seguinte forma:
    - **Ruim:** Vinhos com nota de qualidade 4 ou inferior.
    - **Médio:** Vinhos com nota de qualidade 5 ou 6.
    - **Excelente:** Vinhos com nota de qualidade 7 ou superior.
    """)
    
    df_cat_viz = df.copy()
    def categorize_quality_viz(quality):
        if quality <= 4: return 'Ruim'
        elif quality <= 6: return 'Médio'
        else: return 'Excelente'
    df_cat_viz['categoria_qualidade'] = df_cat_viz['quality'].apply(categorize_quality_viz)
    st.write("**Distribuição das Novas Categorias:**")
    fig_cat_dist = px.histogram(df_cat_viz, x='categoria_qualidade', color_discrete_sequence=['#4B2245'],
                                  category_orders={"categoria_qualidade": ["Ruim", "Médio", "Excelente"]})
    st.plotly_chart(fig_cat_dist, use_container_width=True)
    
    accuracy_cat, report_cat_dict, feature_importance_cat, y_test_cat, y_pred_cat = train_model_2(df)
    st.metric(label="Acurácia no Teste", value=f"{accuracy_cat:.2%}")
    plot_classification_report(report_cat_dict, "Relatório de Classificação")
    plot_confusion_matrix(y_test_cat, y_pred_cat, "Matriz de Confusão", ['Ruim', 'Médio', 'Excelente'])
    plot_feature_importance(feature_importance_cat, "Importância das Características")

elif page == "Conclusões e Comparativo":
    st.title("Conclusões e Comparativo dos Modelos")
    st.markdown("Aqui comparamos os resultados dos dois modelos para extrair os insights finais.")
    
    accuracy1, _, feature_importance1, _, _, _ = train_model_1(df)
    accuracy2, _, feature_importance2, _, _ = train_model_2(df)

    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Modelo Preditivo")
        st.markdown(f"""
        <div style="height: 62px;">
            <div style="color: #808495; font-size: 0.875rem;">Acurácia</div>
            <div style="font-size: 1.75rem; font-weight: 600;">{accuracy1:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("**Top 3 Características:**")
        st.dataframe(feature_importance1.sort_values('importance', ascending=False).head(3))

    with col2:
        st.header("Modelo Otimizado")
        delta_value = accuracy2 - accuracy1
        st.markdown(f"""
        <div style="height: 62px;">
            <div style="color: #808495; font-size: 0.875rem;">Acurácia</div>
            <div style="display: flex; align-items: baseline;">
                <div style="font-size: 1.75rem; font-weight: 600; padding-right: 0.5rem;">{accuracy2:.2%}</div>
                <div style="color: #3C9A00; font-size: 1rem; font-weight: 600;">▲ {delta_value:.2%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**Top 3 Características:**")
        st.dataframe(feature_importance2.sort_values('importance', ascending=False).head(3))
        
    st.markdown("---")
    st.header("Insights Finais")
    
    st.info("""
    **Insight 1: O Salto de Performance**
    
    O agrupamento das classes de qualidade em 'Ruim', 'Médio' e 'Excelente' resultou em um **aumento drástico na acurácia**, saltando de **~69% para ~90%**. Isso prova que simplificar o problema para o modelo, tornando-o mais balanceado e com classes mais distintas, é uma estratégia extremamente eficaz.
    """)
    
    st.info("""
    **Insight 2: A Mudança na Importância das Características**
    
    No modelo preditivo original, o **álcool** era o fator mais importante para diferenciar notas muito próximas (como 5 de 6). No modelo otimizado, os **sulfatos** e a **acidez volátil** ganharam destaque. Isso sugere que esses componentes são fortes indicadores para separar os vinhos nos extremos (especialmente os 'Ruins' dos 'Excelentes'), enquanto o álcool é mais útil para um ajuste fino da qualidade.
    """)
    
    st.info("""
    **Conclusão Final: Qual Modelo Usar?**
    
    - **Use o modelo preditivo** se precisar de uma estimativa da nota *exata* e aceitar uma margem de erro maior.
    - **Use o modelo otimizado** para uma classificação de negócio muito mais confiável e robusta. Para a maioria das aplicações práticas (ex: separar vinhos em prateleiras 'padrão' e 'premium'), o modelo otimizado é a escolha superior.
    """)
