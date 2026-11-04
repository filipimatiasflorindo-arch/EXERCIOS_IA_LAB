import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CARREGAMENTO DOS CONJUNTOS DE DADOS
# ==============================================================================
df_train = pd.read_csv('conjunto_treinamento.csv')
df_val = pd.read_csv('conjunto_validacao.csv')

features = ['x1', 'x2', 'x3']
target = 'd'

X_train = df_train[features].values
d_train = df_train[target].values

X_val = df_val[features].values
d_val = df_val[target].values

P = X_train.shape[0]        # 200 amostras de treino
N = X_train.shape[1]        # 3 sinais de entrada
N1 = 10                     # 10 neurônios na camada escondida
P_val = X_val.shape[0]      # 20 amostras de teste/validação

# ==============================================================================
# 2. FUNÇÕES DE ATIVAÇÃO
# ==============================================================================
def sigmoid(u):
    """Função de ativação logística."""
    return 1.0 / (1.0 + np.exp(-u))

def d_sigmoid(y):
    """Derivada da função sigmóide expressa em função da saída y."""
    return y * (1.0 - y)

# ==============================================================================
# 3. ALGORITMO BACKPROPAGATION (REGRA DELTA GENERALIZADA)
# ==============================================================================
def treinar_rede(seed, eta=0.1, epsilon=1e-6, max_epocas=5000):
    np.random.seed(seed)
    
    # 1. Inicialização dos pesos com valores aleatórios entre 0 e 1
    # W1: camada de entrada -> camada escondida (dimensão 10 x 3)
    # W2: camada escondida -> camada de saída (dimensão 1 x 10)
    W1 = np.random.uniform(0.0, 1.0, (N1, N))
    W2 = np.random.uniform(0.0, 1.0, (1, N1))
    
    historico_eqm = []
    
    for epoca in range(max_epocas):
        # Treinamento padrão a padrão
        for i in range(P):
            x_p = X_train[i:i+1].T  # Vetor coluna (3 x 1)
            d_p = d_train[i]
            
            # --- Passo Forward ---
            u1 = W1 @ x_p
            y1 = sigmoid(u1)        # Saída dos 10 neurônios intermediários (10 x 1)
            
            u2 = W2 @ y1
            y2 = sigmoid(u2)[0, 0]  # Saída final estimada (escalar)
            
            erro = d_p - y2
            
            # --- Passo Backward ---
            # Gradiente local da camada de saída
            delta_o = erro * d_sigmoid(y2)
            
            # Gradiente local da camada intermediária: (10 x 1)
            delta_h = delta_o * (W2.T) * d_sigmoid(y1)
            
            # Atualização das matrizes de pesos
            W2 += eta * delta_o * y1.T
            W1 += eta * (delta_h @ x_p.T)
        
        # --- Cálculo do Erro Quadrático Médio (EQM) da época ---
        y1_todos = sigmoid(W1 @ X_train.T)
        y2_todos = sigmoid(W2 @ y1_todos).ravel()
        eqm = np.mean((d_train - y2_todos) ** 2)
        historico_eqm.append(eqm)
        
        # Critério de parada: estabilização do erro dentro da precisão epsilon
        if epoca > 0 and abs(historico_eqm[-1] - historico_eqm[-2]) <= epsilon:
            break
            
    return epoca + 1, historico_eqm[-1], historico_eqm, W1, W2

# ==============================================================================
# 4. EXECUÇÃO DOS 5 TREINAMENTOS (T1 A T5)
# ==============================================================================
seeds = [10, 42, 100, 2026, 7]
resultados_treino = {}

for idx, seed in enumerate(seeds, start=1):
    nome = f"T{idx}"
    epocas, eqm_final, hist, W1, W2 = treinar_rede(seed=seed, eta=0.1, epsilon=1e-6)
    
    # Validação com as 20 amostras de teste
    y1_val = sigmoid(W1 @ X_val.T)
    y2_val = sigmoid(W2 @ y1_val).ravel()
    
    # Erro relativo percentual amostra a amostra: |d - y| / d * 100
    erros_rel = np.abs(d_val - y2_val) / d_val * 100.0
    erm = np.mean(erros_rel)
    var_erm = np.var(erros_rel, ddof=1)
    
    resultados_treino[nome] = {
        'epocas': epocas,
        'eqm': eqm_final,
        'historico': hist,
        'predicoes_val': y2_val,
        'erm': erm,
        'var_erm': var_erm
    }

# ==============================================================================
# 5. TABELA DO ITEM 2: RESULTADOS FINAIS DOS TREINAMENTOS
# ==============================================================================
tabela_item_2 = pd.DataFrame([
    {
        'Treinamento': f"{i}º ({nome})",
        'Erro Quadrático Médio': f"{resultados_treino[nome]['eqm']:.6e}",
        'Número de Épocas': resultados_treino[nome]['epocas']
    }
    for i, nome in enumerate(resultados_treino.keys(), start=1)
])

print("=" * 65)
print("TABELA ITEM 2: RESULTADOS DOS TREINAMENTOS")
print("=" * 65)
print(tabela_item_2.to_string(index=False))

# ==============================================================================
# 6. TABELA DO ITEM 5: VALIDAÇÃO NO CONJUNTO DE TESTE
# ==============================================================================
colunas_validacao = {
    'Amostra': np.arange(1, P_val + 1),
    'x1': X_val[:, 0],
    'x2': X_val[:, 1],
    'x3': X_val[:, 2],
    'd': d_val
}

for nome in resultados_treino.keys():
    colunas_validacao[f'Vrede ({nome})'] = np.round(resultados_treino[nome]['predicoes_val'], 4)

tabela_item_5 = pd.DataFrame(colunas_validacao)

print("\n" + "=" * 90)
print("TABELA ITEM 5: VALIDAÇÃO COM O CONJUNTO DE TESTE")
print("=" * 90)
print(tabela_item_5.to_string(index=False))

print("\n" + "-" * 90)
resumo_metricas = pd.DataFrame({
    'Métrica': ['Erro Relativo Médio (%)', 'Variância (%)'],
    'T1': [f"{resultados_treino['T1']['erm']:.2f}%", f"{resultados_treino['T1']['var_erm']:.2f}%"],
    'T2': [f"{resultados_treino['T2']['erm']:.2f}%", f"{resultados_treino['T2']['var_erm']:.2f}%"],
    'T3': [f"{resultados_treino['T3']['erm']:.2f}%", f"{resultados_treino['T3']['var_erm']:.2f}%"],
    'T4': [f"{resultados_treino['T4']['erm']:.2f}%", f"{resultados_treino['T4']['var_erm']:.2f}%"],
    'T5': [f"{resultados_treino['T5']['erm']:.2f}%", f"{resultados_treino['T5']['var_erm']:.2f}%"],
})
print(resumo_metricas.to_string(index=False))

# ==============================================================================
# 7. GRÁFICOS DO ITEM 3: OS DOIS TREINAMENTOS COM MAIOR NÚMERO DE ÉPOCAS
# ==============================================================================
maiores_epocas = sorted(resultados_treino.items(), key=lambda item: item[1]['epocas'], reverse=True)[:2]
t_top1_nome, t_top1_data = maiores_epocas[0]
t_top2_nome, t_top2_data = maiores_epocas[1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

ax1.plot(t_top1_data['historico'], color='#1f77b4', lw=1.5)
ax1.set_title(f'Convergência do Treinamento {t_top1_nome} ({t_top1_data["epocas"]} épocas)', fontsize=12)
ax1.set_ylabel('EQM')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(t_top2_data['historico'], color='#d62728', lw=1.5)
ax2.set_title(f'Convergência do Treinamento {t_top2_nome} ({t_top2_data["epocas"]} épocas)', fontsize=12)
ax2.set_xlabel('Época')
ax2.set_ylabel('EQM')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()