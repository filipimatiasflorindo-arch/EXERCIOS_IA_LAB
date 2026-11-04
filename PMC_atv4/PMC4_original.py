import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. Leitura e Preparação da Série Temporal a partir dos Ficheiros CSV
# -------------------------------------------------------------------------
def carregar_dados():
  # Permite flexibilidade caso os ficheiros se chamem com ou sem '_2'
  caminho_treino = (
      'treinamento_2.csv'
      if os.path.exists('treinamento_2.csv')
      else 'treinamento.csv'
  )
  caminho_teste = (
      'teste_2.csv' if os.path.exists('teste_2.csv') else 'teste.csv'
  )

  df_train = pd.read_csv(caminho_treino)
  df_test = pd.read_csv(caminho_teste)

  # Limpeza dos nomes das colunas (removendo eventuais espaços em branco)
  df_train.columns = df_train.columns.str.strip()
  df_test.columns = df_test.columns.str.strip()

  # Extração dos valores da série f(t) como listas de números reais
  serie_treino = df_train['f(t)'].astype(float).tolist()  # t = 1 .. 100
  serie_teste = df_test['f(t)'].astype(float).tolist()  # t = 101 .. 120

  # Série temporal contínua e ordenada (t = 1 .. 120)
  serie_completa = serie_treino + serie_teste
  return serie_completa


# -------------------------------------------------------------------------
# 2. Estruturação dos Padrões TDNN (Janelas de Atraso Temporal)
# -------------------------------------------------------------------------
def construir_janelas_tdnn(serie, p, t_inicio, t_fim):
  """Constrói os vetores de entrada [x(t-1), x(t-2), ..., x(t-p)]

  e o respetivo valor desejado d(t) = x(t).
  Nota: t é indexado a 1 (t=1 corresponde ao índice 0 da lista).
  """
  X, y = [], []
  for t in range(t_inicio, t_fim + 1):
    # Janela temporal: do atraso 1 até p (ordem decrescente de tempo)
    janela = [serie[(t - 1) - atraso] for atraso in range(1, p + 1)]
    X.append(janela)
    y.append([serie[t - 1]])
  return np.array(X), np.array(y)


# -------------------------------------------------------------------------
# 3. Funções de Ativação e Algoritmo de Treino MLP (Backprop com Momentum)
# -------------------------------------------------------------------------
def sigmoide(v):
  return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))


def derivada_sigmoide(y):
  return y * (1.0 - y)


def treinar_tdnn(
    X,
    d,
    p,
    n_ocultos,
    semente,
    eta=0.1,
    alpha=0.8,
    epsilon=0.5e-6,
    max_epocas=10000,
    valor_bias=-1,
):
  np.random.seed(semente)

  # Inicialização das matrizes de pesos com valores aleatórios entre 0 e 1
  # Camada oculta: n_ocultos neurónios x (p entradas + 1 bias)
  # Camada de saída: 1 neurónio x (n_ocultos entradas + 1 bias)
  W_h = np.random.uniform(0.0, 1.0, (n_ocultos, p + 1))
  W_o = np.random.uniform(0.0, 1.0, (1, n_ocultos + 1))

  # Armazenamento das variações de pesos da iteração anterior para o termo momentum
  dW_h_ant = np.zeros_like(W_h)
  dW_o_ant = np.zeros_like(W_o)

  N = len(X)
  historico_eqm = []

  for epoca in range(max_epocas):
    eqm_acumulado = 0.0

    # Atualização padrão a padrão
    for k in range(N):
      # --- Passo Forward ---
      x_k = np.insert(X[k], 0, valor_bias)
      net_h = W_h @ x_k
      y_h = sigmoide(net_h)

      y_h_bias = np.insert(y_h, 0, valor_bias)
      net_o = W_o @ y_h_bias
      y_o = sigmoide(net_o)

      # Erro de estimação
      e_k = d[k] - y_o
      eqm_acumulado += np.sum(e_k**2)

      # --- Passo Backward ---
      delta_o = e_k * derivada_sigmoide(y_o)
      delta_h = (W_o[:, 1:].T @ delta_o) * derivada_sigmoide(y_h)

      # Atualização dos pesos com Momentum
      dW_o = eta * np.outer(delta_o, y_h_bias) + alpha * dW_o_ant
      dW_h = eta * np.outer(delta_h, x_k) + alpha * dW_h_ant

      W_o += dW_o
      W_h += dW_h

      dW_o_ant = dW_o
      dW_h_ant = dW_h

    # Erro Quadrático Médio da época
    eqm_epoca = eqm_acumulado / (2.0 * N)
    historico_eqm.append(eqm_epoca)

    # Critério de paragem por variação de precisão epsilon
    if epoca > 0 and abs(historico_eqm[-1] - historico_eqm[-2]) < epsilon:
      break
    if eqm_epoca < epsilon:
      break

  return W_h, W_o, historico_eqm


def prever_tdnn(X, W_h, W_o, valor_bias=-1):
  previsoes = []
  for k in range(len(X)):
    x_k = np.insert(X[k], 0, valor_bias)
    y_h = sigmoide(W_h @ x_k)
    y_h_bias = np.insert(y_h, 0, valor_bias)
    y_o = sigmoide(W_o @ y_h_bias)
    previsoes.append(y_o[0])
  return np.array(previsoes)


# -------------------------------------------------------------------------
# 4. Execução dos Treinos e Consolidação dos Resultados
# -------------------------------------------------------------------------
if __name__ == '__main__':
  serie_temporal = carregar_dados()

  topologias = {
      'Rede 1': {'p': 5, 'N1': 10, 'sementes': [10, 20, 30]},
      'Rede 2': {'p': 10, 'N1': 15, 'sementes': [40, 50, 60]},
      'Rede 3': {'p': 15, 'N1': 25, 'sementes': [70, 80, 90]},
  }

  resultados = {}

  for nome_rede, conf in topologias.items():
    p, N1 = conf['p'], conf['N1']
    # Treino: t vai de p+1 até 100
    X_tr, y_tr = construir_janelas_tdnn(
        serie_temporal, p, t_inicio=p + 1, t_fim=100
    )
    # Teste: t vai de 101 até 120
    X_te, y_te = construir_janelas_tdnn(
        serie_temporal, p, t_inicio=101, t_fim=120
    )
    d_teste = y_te.flatten()

    resultados[nome_rede] = []

    for i, semente in enumerate(conf['sementes']):
      W_h, W_o, hist = treinar_tdnn(
          X_tr,
          y_tr,
          p,
          N1,
          semente,
          eta=0.1,
          alpha=0.8,
          epsilon=0.5e-6,
      )
      previsoes = prever_tdnn(X_te, W_h, W_o)

      # Erro relativo percentual: |d - y| / d * 100
      erros_rel = np.abs(d_teste - previsoes) / d_teste * 100.0
      erro_medio = np.mean(erros_rel)
      variancia = np.var(erros_rel, ddof=0)

      resultados[nome_rede].append({
          'treino': f'T{i + 1}',
          'eqm_final': hist[-1],
          'epocas': len(hist),
          'historico': hist,
          'previsoes': previsoes,
          'erro_medio': erro_medio,
          'variancia': variancia,
      })

  # Identificação da melhor execução de cada topologia
  melhores = {'Rede 1': 2, 'Rede 2': 2, 'Rede 3': 2}  # Correspondente a T3

  # Gráficos de EQM (Item 4)
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
  for idx, (nome, id_run) in enumerate(melhores.items()):
    info = resultados[nome][id_run]
    axes[idx].plot(info['historico'], color='#2563eb', lw=1.6)
    axes[idx].set_title(
        f"{nome} - {info['treino']} (Melhor Treino)\nÉpocas: {info['epocas']} |"
        f" EQM: {info['eqm_final']:.6f}"
    )
    axes[idx].set_xlabel('Épocas')
    axes[idx].set_ylabel('EQM')
    axes[idx].grid(True, linestyle='--', alpha=0.6)
  plt.tight_layout()
  plt.savefig('grafico_eqm_topologias.png', dpi=300)
  plt.close()

  # Gráficos de Estimação vs Desejado (Item 5)
  t_eixo = np.arange(101, 121)
  d_alvo = serie_temporal[100:120]

  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
  for idx, (nome, id_run) in enumerate(melhores.items()):
    info = resultados[nome][id_run]
    axes[idx].plot(
        t_eixo, d_alvo, 'o--', color='black', label='Desejado f(t)', lw=1.5
    )
    axes[idx].plot(
        t_eixo,
        info['previsoes'],
        's-',
        color='#dc2626',
        label=f"Estimado ({info['treino']})",
        lw=1.5,
    )
    axes[idx].set_title(
        f"{nome} - {info['treino']}\nErro Relativo Médio:"
        f" {info['erro_medio']:.2f}%"
    )
    axes[idx].set_xlabel('Instante de Tempo (t)')
    axes[idx].set_ylabel('f(t)')
    axes[idx].set_xticks(t_eixo[::2])
    axes[idx].legend()
    axes[idx].grid(True, linestyle='--', alpha=0.6)
  plt.tight_layout()
  plt.savefig('grafico_estimacao_topologias.png', dpi=300)
  plt.close()