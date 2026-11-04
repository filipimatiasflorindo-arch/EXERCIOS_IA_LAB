import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. FUNÇÕES DE ATIVAÇÃO E AUXILIARES
# ==============================================================================

def sigmoid(u):
    """
    Função de ativação logística (sigmoide): f(u) = 1 / (1 + e^(-u))
    Utiliza np.clip para evitar overflow numérico.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500.0, 500.0)))


def compute_eqm_vectorized(X_bias, d, W_h, W_o, bias_val=-1.0):
  """Calcula o Erro Quadrático Médio (EQM) de forma totalmente vetorizada

  sobre todo o conjunto de dados (feedforward matricial em lote).
  EQM = (1 / (2 * N)) * sum_{k=1}^N sum_{j=1}^m (d_kj - y_kj)^2
  """
  net_h = X_bias @ W_h.T
  y_h = sigmoid(net_h)
  y_h_bias = np.column_stack([np.full(len(X_bias), bias_val), y_h])
  net_o = y_h_bias @ W_o.T
  y_o = sigmoid(net_o)
  erro = d - y_o
  return np.sum(erro * erro) / (2.0 * len(X_bias))


# ==============================================================================
# 2. ALGORITMO DE TREINAMENTO MLP OTIMIZADO (BACKPROPAGATION & MOMENTUM)
# ==============================================================================


def train_mlp(
    X,
    d,
    W_h_init,
    W_o_init,
    eta=0.1,
    alpha=0.0,
    epsilon=1e-6,
    max_epochs=10000,
    bias_val=-1.0,
):
  """Executa o treinamento da rede MLP utilizando aprendizado padrão a padrão.

  Otimizado com pré-alocação de memória e cálculo vetorizado do EQM.

  Parâmetros:
    X: Matriz de entrada (N x p)
    d: Matriz de saídas desejadas (N x m)
    W_h_init: Matriz de pesos inicial da camada oculta (15 x 5)
    W_o_init: Matriz de pesos inicial da camada de saída (3 x 16)
    eta: Taxa de aprendizado (padrão = 0.1)
    alpha: Fator de momentum (0.0 para padrão, 0.9 para com momentum)
    epsilon: Precisão para o critério de parada (10^-6)
    max_epochs: Número máximo de épocas
    bias_val: Valor fixo atribuído ao bias (padrão = -1.0)
  """
  N = len(X)
  num_hidden = W_h_init.shape[0]

  # Pré-alocação da matriz com a coluna de bias inserida (elimina np.insert dentro do loop)
  X_bias = np.column_stack([np.full(N, bias_val), X])

  # Cópias de trabalho das matrizes de pesos
  W_h = W_h_init.copy()
  W_o = W_o_init.copy()

  # Matrizes de variação anterior para o termo de momentum: Delta W(t-1)
  dW_h_prev = np.zeros_like(W_h)
  dW_o_prev = np.zeros_like(W_o)

  # Buffer estático para a camada oculta com bias
  y_h_bias = np.empty(num_hidden + 1)
  y_h_bias[0] = bias_val

  # Histórico do EQM
  eqm_history = [compute_eqm_vectorized(X_bias, d, W_h, W_o, bias_val)]

  start_time = time.time()

  for epoch in range(1, max_epochs + 1):
    for k in range(N):
      x_k = X_bias[k]

      # --- Passo Forward ---
      net_h = W_h @ x_k
      y_h = sigmoid(net_h)
      y_h_bias[1:] = y_h

      net_o = W_o @ y_h_bias
      y_o = sigmoid(net_o)

      # Erro instantâneo no padrão k
      e_k = d[k] - y_o

      # --- Passo Backward (Gradientes Locais) ---
      # Gradiente local da camada de saída: delta_o = e_k * y_o * (1 - y_o)
      delta_o = e_k * (y_o * (1.0 - y_o))

      # Gradiente local da camada oculta: delta_h = (W_o^T * delta_o) * y_h * (1 - y_h)
      # Nota: W_o[:, 1:] ignora os pesos associados ao bias da camada oculta
      delta_h = (W_o[:, 1:].T @ delta_o) * (y_h * (1.0 - y_h))

      # --- Atualização dos Pesos Sinápticos ---
      # dW(t) = eta * delta * entrada + alpha * dW(t-1)
      dW_o = eta * np.outer(delta_o, y_h_bias) + alpha * dW_o_prev
      dW_h = eta * np.outer(delta_h, x_k) + alpha * dW_h_prev

      W_o += dW_o
      W_h += dW_h

      dW_o_prev = dW_o
      dW_h_prev = dW_h

    # Avaliação do EQM da época
    cur_eqm = compute_eqm_vectorized(X_bias, d, W_h, W_o, bias_val)
    eqm_history.append(cur_eqm)

    # Critério de convergência: |EQM(k) - EQM(k-1)| <= epsilon ou EQM <= epsilon
    if abs(cur_eqm - eqm_history[-2]) <= epsilon or cur_eqm <= epsilon:
      break

  elapsed_time = time.time() - start_time
  return W_h, W_o, eqm_history, elapsed_time


# ==============================================================================
# 3. PÓS-PROCESSAMENTO E VALIDAÇÃO NO CONJUNTO DE TESTE (ITENS 3 E 4)
# ==============================================================================


def evaluate_test_set(X, d, W_h, W_o, bias_val=-1.0):
  """Executa a inferência para o conjunto de teste, aplica o critério do

  arredondamento simétrico (Item 3) e calcula a taxa de acerto (Item 4).
  """
  N = len(X)
  num_hidden = W_h.shape[0]

  X_bias = np.column_stack([np.full(N, bias_val), X])
  y_h_bias = np.empty(num_hidden + 1)
  y_h_bias[0] = bias_val

  correct = 0
  rows = []

  for k in range(N):
    x_k = X_bias[k]
    y_h_bias[1:] = sigmoid(W_h @ x_k)
    y_o = sigmoid(W_o @ y_h_bias)

    # Critério de arredondamento simétrico: y_pos = 1 se y >= 0.5; 0 se y < 0.5
    y_pos = np.where(y_o >= 0.5, 1, 0)
    is_correct = np.array_equal(y_pos, d[k])
    if is_correct:
      correct += 1

    rows.append({
        "Amostra": k + 1,
        "x1": X[k, 0],
        "x2": X[k, 1],
        "x3": X[k, 2],
        "x4": X[k, 3],
        "d1": int(d[k, 0]),
        "d2": int(d[k, 1]),
        "d3": int(d[k, 2]),
        "y1_pos": int(y_pos[0]),
        "y2_pos": int(y_pos[1]),
        "y3_pos": int(y_pos[2]),
        "y1": y_o[0],
        "y2": y_o[1],
        "y3": y_o[2],
        "Correto": is_correct,
    })

  accuracy = (correct / N) * 100.0
  df_results = pd.DataFrame(rows)
  return df_results, accuracy


def print_formatted_table(df_results, accuracy):
  """Imprime a tabela do Item 4 formatada exatamente conforme o modelo do PDF."""
  print("\n" + "=" * 116)
  print(
      f"{'Amostra':^7} | {'x1':^6} {'x2':^6} {'x3':^6} {'x4':^6} | {'d1':^2}"
      f" {'d2':^2} {'d3':^2} | {'y1_pós':^6} {'y2_pós':^6} {'y3_pós':^6} |"
      f" {'y1':^8} {'y2':^8} {'y3':^8} | {'Acerto':^6}"
  )
  print("=" * 116)

  for _, row in df_results.iterrows():
    acerto_str = "SIM" if row["Correto"] else "NÃO"
    print(
        f"{int(row['Amostra']):^7d} | {row['x1']:.4f} {row['x2']:.4f}"
        f" {row['x3']:.4f} {row['x4']:.4f} | {int(row['d1']):^2d}"
        f" {int(row['d2']):^2d} {int(row['d3']):^2d} | {int(row['y1_pos']):^6d}"
        f" {int(row['y2_pos']):^6d} {int(row['y3_pos']):^6d} |"
        f" {row['y1']:.4f}   {row['y2']:.4f}   {row['y3']:.4f}  |"
        f" {acerto_str:^6}"
    )

  print("=" * 116)
  print(
      f"Taxa de Acerto: {df_results['Correto'].sum()}/{len(df_results)}"
      f" ({accuracy:.2f}%)\n"
  )


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
  # Leitura das bases de treinamento e teste
  train_df = pd.read_csv("treinamento.csv")
  test_df = pd.read_csv("teste.csv")

  X_train = train_df[["x1", "x2", "x3", "x4"]].values
  d_train = train_df[["d1", "d2", "d3"]].values

  X_test = test_df[["x1", "x2", "x3", "x4"]].values
  d_test = test_df[["d1", "d2", "d3"]].values

  # Fixação da semente para reprodutibilidade das matrizes iniciais
  np.random.seed(42)

  # Inicialização das matrizes de pesos com valores aleatórios entre 0 e 1:
  # - Camada oculta: 15 neurônios x (4 entradas + 1 bias) = 15 x 5
  # - Camada de saída: 3 neurônios x (15 neurônios ocultos + 1 bias) = 3 x 16
  W_h_init = np.random.uniform(0.0, 1.0, (15, 5))
  W_o_init = np.random.uniform(0.0, 1.0, (3, 16))

  # --------------------------------------------------------------------------
  # ITEM 1: Backpropagation Padrão (eta = 0.1, alpha = 0.0, epsilon = 10^-6)
  # --------------------------------------------------------------------------
  print("=" * 80)
  print("ITEM 1: Treinamento Backpropagation Padrão (eta=0.1, epsilon=1e-6)")
  print("=" * 80)
  W_h_std, W_o_std, eqm_std, t_std = train_mlp(
      X_train, d_train, W_h_init, W_o_init, eta=0.1, alpha=0.0, epsilon=1e-6
  )
  print(f"Épocas até convergência: {len(eqm_std) - 1}")
  print(f"Tempo de processamento:  {t_std:.2f} segundos")
  print(f"EQM inicial:             {eqm_std[0]:.6f}")
  print(f"EQM final:               {eqm_std[-1]:.6f}\n")

  # --------------------------------------------------------------------------
  # ITEM 2: Backpropagation com Momentum (eta = 0.1, alpha = 0.9, epsilon = 10^-6)
  # Utilizando as MESMAS matrizes de pesos iniciais do item 1
  # --------------------------------------------------------------------------
  print("=" * 80)
  print(
      "ITEM 2: Treinamento Backpropagation com Momentum (eta=0.1, alpha=0.9,"
      " epsilon=1e-6)"
  )
  print("=" * 80)
  W_h_mom, W_o_mom, eqm_mom, t_mom = train_mlp(
      X_train, d_train, W_h_init, W_o_init, eta=0.1, alpha=0.9, epsilon=1e-6
  )
  print(f"Épocas até convergência: {len(eqm_mom) - 1}")
  print(f"Tempo de processamento:  {t_mom:.2f} segundos")
  print(f"EQM inicial:             {eqm_mom[0]:.6f}")
  print(f"EQM final:               {eqm_mom[-1]:.6f}\n")

  # --------------------------------------------------------------------------
  # GRÁFICOS DO EQM EM FUNÇÃO DAS ÉPOCAS (NÃO SUPERPOSTOS NA MESMA FOLHA)
  # --------------------------------------------------------------------------
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))

  # Gráfico 1: Padrão
  axes[0].plot(eqm_std, color="#1f77b4", lw=2, label="Backpropagation Padrão")
  axes[0].set_title(
      f"Backpropagation Padrão (η = 0.1)\nÉpocas: {len(eqm_std)-1} | Tempo:"
      f" {t_std:.2f}s | EQM Final: {eqm_std[-1]:.6f}",
      fontsize=11,
      fontweight="bold",
  )
  axes[0].set_xlabel("Época de Treinamento", fontsize=10)
  axes[0].set_ylabel("Erro Quadrático Médio (EQM)", fontsize=10)
  axes[0].grid(True, linestyle="--", alpha=0.6)
  axes[0].legend(loc="upper right")

  # Gráfico 2: Com Momentum
  axes[1].plot(
      eqm_mom, color="#ff7f0e", lw=2, label="Backpropagation com Momentum"
  )
  axes[1].set_title(
      f"Backpropagation com Momentum (η = 0.1, α = 0.9)\nÉpocas:"
      f" {len(eqm_mom)-1} | Tempo: {t_mom:.2f}s | EQM Final:"
      f" {eqm_mom[-1]:.6f}",
      fontsize=11,
      fontweight="bold",
  )
  axes[1].set_xlabel("Época de Treinamento", fontsize=10)
  axes[1].set_ylabel("Erro Quadrático Médio (EQM)", fontsize=10)
  axes[1].grid(True, linestyle="--", alpha=0.6)
  axes[1].legend(loc="upper right")

  plt.tight_layout()
  plot_path = "graficos_eqm.png"
  plt.savefig(plot_path, dpi=300)
  print(f"Gráficos salvos com sucesso em '{plot_path}'.\n")

  # --------------------------------------------------------------------------
  # ITENS 3 E 4: PÓS-PROCESSAMENTO E VALIDAÇÃO NO CONJUNTO DE TESTE
  # --------------------------------------------------------------------------
  print("=" * 80)
  print("ITENS 3 e 4: Pós-processamento Simétrico e Tabela de Validação")
  print("=" * 80)
  df_tabela, taxa_acerto = evaluate_test_set(X_test, d_test, W_h_mom, W_o_mom)
  print_formatted_table(df_tabela, taxa_acerto)

  # Salva a tabela com os resultados preenchidos em CSV
  df_tabela.to_csv("tabela_teste_preenchida.csv", index=False)
  print("Tabela preenchida salva em 'tabela_teste_preenchida.csv'.")