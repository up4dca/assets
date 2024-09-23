import cupy as cp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import StringIO

# Paleta de cores neon escuras
neon_palette = ['#00FFFF', '#00FF00', '#FF00FF', '#FFFF00', '#008080', '#008000', '#800080', '#808000']

# Dados Sintéticos Realistas (Binance) - Adaptado para leitura de CSV
binance_csv = """Altcoin,Preço (USD),Volume (24h),Market Cap (USD),Variação (24h)
Aergo (AERGO),0.12,5000000,60000000,2.5
Aion (AION),0.08,3000000,40000000,-1.2
Bluzelle (BLZ),0.05,2000000,25000000,0.8
Celer Network (CELR),0.03,1500000,18000000,1.5
COTI (COTI),0.04,1000000,20000000,-0.5
Elrond (EGLD),50,8000000,2500000000,3.2
Fetch.ai (FET),0.25,6000000,125000000,1.8
Harmony (ONE),0.02,4000000,10000000,-0.9
Hedera Hashgraph (HBAR),0.06,2500000,30000000,0.7
ICON (ICX),0.30,7000000,150000000,2.1
IOST (IOST),0.01,1800000,50000000,-1.5
Kava (KAVA),0.80,3500000,400000000,0.5
Komodo (KMD),0.55,1500000,275000000,-0.2
Kyber Network (KNC),0.70,1000000,350000000,3.0
Nervos Network (CKB),0.008,8000000,40000000,-0.7
Ocean Protocol (OCEAN),0.45,1200000,225000000,1.8
Ontology (ONT),0.50,2000000,250000000,1.2
Quant (QNT),100,4000000,5000000000,-1.5
Ravencoin (RVN),0.035,3000000,175000000,2.1
Reserve (RSV),0.02,1500000,10000000,0.5"""

df_altcoins_binance = pd.read_csv(StringIO(binance_csv))

# Dados (GitHub)
github_csv = """Altcoin,Número de Commits (último mês),Número de Issues Abertas,Número de Pull Requests (último mês),Atividade da Comunidade (escala 1-10)
Aergo (AERGO),180,30,60,7.5
Aion (AION),150,25,50,6.8
Bluzelle (BLZ),120,20,40,6.2
Celer Network (CELR),190,32,65,7.8
COTI (COTI),160,28,55,7.2
Elrond (EGLD),250,40,80,8.5
Fetch.ai (FET),220,35,70,8.2
Harmony (ONE),170,30,58,7.5
Hedera Hashgraph (HBAR),200,38,75,8.0
ICON (ICX),230,35,75,8.3
IOST (IOST),140,22,45,6.5
Kava (KAVA),210,35,70,8.0
Komodo (KMD),165,28,55,7.3
Kyber Network (KNC),240,40,80,8.7
Nervos Network (CKB),185,32,62,7.8
Ocean Protocol (OCEAN),205,35,70,8.2
Ontology (ONT),195,33,68,7.9
Quant (QNT),260,45,90,9.0
Ravencoin (RVN),175,30,60,7.6
Reserve (RSV),155,25,50,7.0"""

df_altcoins_github = pd.read_csv(StringIO(github_csv))

# Função para plotar gráfico de caixa e violino
def plot_box_violin(df, x_col, y_col, title):
    fig = go.Figure()
    for i, val in enumerate(df[x_col].unique()):
        fig.add_trace(go.Violin(x=df[x_col][df[x_col] == val],
                                 y=df[y_col][df[x_col] == val],
                                 name=val,
                                 box_visible=True,
                                 meanline_visible=True,
                                 line_color=neon_palette[i % len(neon_palette)]))
    fig.update_layout(title=title,
                      xaxis_title=x_col,
                      yaxis_title=y_col,
                      plot_bgcolor='black',
                      paper_bgcolor='black',
                      font_color='white')
    fig.show()

# Função para plotar gráfico de pizza com sunburst
def plot_sunburst_chart(df, values, names, title):
    fig = px.sunburst(df, values=values, path=[names], title=title,
                      color_discrete_sequence=neon_palette)
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.show()

# Função para plotar boxplot
def plot_boxplot(df, x_col, y_col, title):
    fig = px.box(df, x=x_col, y=y_col, title=title, 
                 color_discrete_sequence=neon_palette)
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.show()

# Função para plotar matriz de correlação
def plot_correlation_matrix(df, title):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                   x=corr_matrix.index,
                                   y=corr_matrix.columns,
                                   colorscale='Viridis'))
    fig.update_layout(title=title,
                      xaxis_nticks=36,
                      plot_bgcolor='black', paper_bgcolor='black', font_color='white')
    fig.show()

# Visualizações dos Dados de Binance
plot_sunburst_chart(df_altcoins_binance, 'Market Cap (USD)', 'Altcoin', 'Market Cap das Altcoins')
plot_boxplot(df_altcoins_binance, 'Altcoin', 'Volume (24h)', 'Distribuição do Volume (24h)')
plot_box_violin(df_altcoins_binance, 'Altcoin', 'Preço (USD)', 'Distribuição do Preço das Altcoins (USD)')
fig = px.scatter(df_altcoins_binance, x="Preço (USD)", y="Volume (24h)", color="Altcoin", 
                 size='Market Cap (USD)', hover_data=['Variação (24h)'],
                 color_discrete_sequence=neon_palette)
fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white', title='Relação Preço-Volume')
fig.show()
fig = px.parallel_coordinates(df_altcoins_binance, color="Variação (24h)", 
                             dimensions=["Preço (USD)", "Volume (24h)", "Market Cap (USD)", "Variação (24h)"],
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=0)
fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white', title='Coordenadas Paralelas')
fig.show()

# **Corrigido:** Usando go.Heatmap para o mapa de calor de densidade
fig = go.Figure(go.Heatmap(
    z=df_altcoins_binance["Market Cap (USD)"],
    x=df_altcoins_binance["Preço (USD)"],
    y=df_altcoins_binance["Volume (24h)"],
    colorscale="Viridis"
))
fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white', title='Mapa de Calor de Densidade')
fig.show()


# Visualizações dos Dados de GitHub
plot_box_violin(df_altcoins_github, 'Altcoin', 'Número de Commits (último mês)', 'Número de Commits por Altcoin')
plot_correlation_matrix(df_altcoins_github, 'Matriz de Correlação - Dados GitHub')
fig = px.parallel_categories(df_altcoins_github, dimensions=['Altcoin', 'Número de Commits (último mês)', 
                                                            'Número de Issues Abertas', 'Número de Pull Requests (último mês)'],
                             color="Atividade da Comunidade (escala 1-10)", color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white', title='Categorias Paralelas - GitHub')
fig.show()

# Exemplo de uso das funções originais (opcional)
# df = generate_altcoin_data()
# plot_time_series(df)
# plot_pca(df)