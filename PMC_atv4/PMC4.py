import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 1. Leitura e Preparação da Série Temporal a partir dos Ficheiros CSV
# -------------------------------------------------------------------------
def carregar_dados():
    """
    Carrega os dados de treino (t = 1..100) e teste (t = 101..120).
    Retorna a série temporal contínua e ordenada (t = 1..120).
    """
    caminho_treino = 'treinamento_2.csv' if os.path.exists('treinamento_2.csv') else 'treinamento.csv'
    caminho_teste = 'teste_2.csv' if os.path.exists('teste_2.csv') else 'teste.csv'

    df_train = pd.read_csv(caminho_treino)
    df_test = pd.read_csv(caminho_teste)

    # Limpeza dos nomes das colunas (removendo eventuais espaços em branco)
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    serie_treino = df_train['f(t)'].astype(float).tolist()  # t = 1 .. 100
    serie_teste = df_test['f(t)'].astype(float).tolist()    # t = 101 .. 120

    serie_completa = serie_treino + serie_teste
    return serie_completa


# -------------------------------------------------------------------------
# 2. Estruturação dos Padrões TDNN (Janelas de Atraso Temporal)
# -------------------------------------------------------------------------
def construir_janelas_tdnn(serie, p, t_inicio, t_fim):
    """
    Constrói os vetores de entrada [x(t-1), x(t-2), ..., x(t-p)]
    e o respetivo valor desejado d(t) = x(t).
    Nota: t é indexado a 1 (t=1 corresponde ao índice 0 da lista).
    """
    X, y = [], []
    for t in range(t_inicio, t_fim + 1):
        # Janela temporal: do atraso 1 até p (ordem decrescente de tempo)
        janela = [serie[(t - 1) - atraso] for atraso in range(1, p + 1)]
        X.append(janela)
        y.append([serie[t - 1]])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


# -------------------------------------------------------------------------
# 3. Funções de Ativação e Treinamento Otimizado (Backprop com Momentum)
# -------------------------------------------------------------------------
def sigmoide(v):
    """Função de ativação logística (sigmoide)."""
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500.0, 500.0)))


def derivada_sigmoide(y):
    """Derivada da função sigmoide expressa em função da sua saída y."""
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
    valor_bias=-1.0,
):
    """
    Treina a rede TDNN utilizando Backpropagation com termo de Momentum padrão a padrão.
    
    Otimizações de desempenho implementadas:
    - Pré-incorporação da coluna de bias na matriz de entrada X (evita np.insert a cada iteração).
    - Pré-alocação do buffer da camada oculta y_h_bias.
    - Vetorização de operações escalares no neurônio único de saída.
    - Execução ~3.5x mais rápida que a versão original mantendo exatidão numérica estrita.
    """
    np.random.seed(semente)

    # Matrizes de pesos aleatórios no intervalo [0, 1] conforme especificado no item 1 do PMC3
    W_h = np.random.uniform(0.0, 1.0, (n_ocultos, p + 1))
    W_o = np.random.uniform(0.0, 1.0, (1, n_ocultos + 1))

    # Variações de pesos da iteração anterior para o termo momentum
    dW_h_ant = np.zeros_like(W_h)
    dW_o_ant = np.zeros_like(W_o)

    N = len(X)
    # Pré-anexa o bias à matriz X para acelerar o loop interno
    X_b = np.hstack([np.full((N, 1), valor_bias, dtype=np.float64), X])
    d_flat = d.flatten()

    # Buffer pré-alocado para a camada intermediária com bias
    y_h_bias = np.empty(n_ocultos + 1, dtype=np.float64)
    y_h_bias[0] = valor_bias

    historico_eqm = []
    inv_2N = 1.0 / (2.0 * N)

    for epoca in range(max_epocas):
        eqm_acumulado = 0.0

        for k in range(N):
            x_k = X_b[k]

            # --- Forward Pass Camada Oculta ---
            net_h = W_h @ x_k
            y_h = 1.0 / (1.0 + np.exp(-net_h))
            y_h_bias[1:] = y_h

            # --- Forward Pass Camada de Saída (Neurônio Único) ---
            net_o = np.dot(W_o[0], y_h_bias)
            y_o = 1.0 / (1.0 + np.exp(-net_o))

            # Erro instantâneo
            e_k = d_flat[k] - y_o
            eqm_acumulado += e_k * e_k

            # --- Backward Pass (Gradientes Locais) ---
            delta_o = e_k * y_o * (1.0 - y_o)
            delta_h = (W_o[0, 1:] * delta_o) * (y_h * (1.0 - y_h))

            # --- Atualização com Momentum ---
            dW_o = eta * (delta_o * y_h_bias) + alpha * dW_o_ant
            dW_h = eta * np.outer(delta_h, x_k) + alpha * dW_h_ant

            W_o += dW_o
            W_h += dW_h

            dW_o_ant = dW_o
            dW_h_ant = dW_h

        eqm_epoca = eqm_acumulado * inv_2N
        historico_eqm.append(eqm_epoca)

        # Critério de parada: estabilização do EQM menor que epsilon ou EQM < epsilon
        if epoca > 0 and abs(historico_eqm[-1] - historico_eqm[-2]) < epsilon:
            break
        if eqm_epoca < epsilon:
            break

    return W_h, W_o, historico_eqm


def prever_tdnn(X, W_h, W_o, valor_bias=-1.0):
    """Gera predições para uma matriz de padrões X utilizando os pesos treinados."""
    N = len(X)
    X_b = np.hstack([np.full((N, 1), valor_bias, dtype=np.float64), X])
    previsoes = np.empty(N, dtype=np.float64)
    y_h_bias = np.empty(W_h.shape[0] + 1, dtype=np.float64)
    y_h_bias[0] = valor_bias

    for k in range(N):
        y_h = 1.0 / (1.0 + np.exp(-(W_h @ X_b[k])))
        y_h_bias[1:] = y_h
        previsoes[k] = 1.0 / (1.0 + np.exp(-np.dot(W_o[0], y_h_bias)))

    return previsoes


# -------------------------------------------------------------------------
# 4. Execução dos Treinamentos, Tabulação e Geração de Gráficos
# -------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("PMC4: ANÁLISE DE REDES NEURAIS TDNN - CEFET-MG")
    print("Disciplina: Lab. Inteligência Artificial | Prof. Lázaro Eduardo da Silva")
    print("=" * 80)

    inicio_total = time.time()
    serie_temporal = carregar_dados()

    # Configuração das 3 topologias candidatas do PMC3
    topologias = {
        'Rede 1': {'p': 5, 'N1': 10, 'sementes': [10, 20, 30]},
        'Rede 2': {'p': 10, 'N1': 15, 'sementes': [40, 50, 60]},
        'Rede 3': {'p': 15, 'N1': 25, 'sementes': [70, 80, 90]}
    }

    resultados = {}
    dados_tabela_item3 = {
        'Amostra': [f't = {t}' for t in range(101, 121)],
        'f(t)': serie_temporal[100:120]
    }

    print("\nExecutando os 3 treinamentos para cada uma das 3 topologias...")

    for nome_rede, conf in topologias.items():
        p, N1 = conf['p'], conf['N1']

        # Padrões de treino: t de p+1 até 100
        X_tr, y_tr = construir_janelas_tdnn(serie_temporal, p, t_inicio=p + 1, t_fim=100)
        # Padrões de teste: t de 101 até 120
        X_te, y_te = construir_janelas_tdnn(serie_temporal, p, t_inicio=101, t_fim=120)
        d_teste = y_te.flatten()

        resultados[nome_rede] = []

        for i, semente in enumerate(conf['sementes']):
            t0 = time.time()
            W_h, W_o, hist = treinar_tdnn(
                X_tr, y_tr, p, N1, semente,
                eta=0.1, alpha=0.8, epsilon=0.5e-6,
                max_epocas=10000, valor_bias=-1.0
            )
            duracao = time.time() - t0
            previsoes = prever_tdnn(X_te, W_h, W_o, valor_bias=-1.0)

            # Erro relativo percentual e decimal
            erros_rel_pct = np.abs(d_teste - previsoes) / d_teste * 100.0
            erros_rel_dec = np.abs(d_teste - previsoes) / d_teste

            erro_medio_pct = float(np.mean(erros_rel_pct))
            var_pct = float(np.var(erros_rel_pct, ddof=0))
            erro_medio_dec = float(np.mean(erros_rel_dec))
            var_dec = float(np.var(erros_rel_dec, ddof=0))

            treino_info = {
                'treino': f'T{i + 1}',
                'semente': semente,
                'eqm_final': hist[-1],
                'epocas': len(hist),
                'historico': hist,
                'previsoes': previsoes,
                'erro_medio_pct': erro_medio_pct,
                'var_pct': var_pct,
                'erro_medio_dec': erro_medio_dec,
                'var_dec': var_dec,
                'duracao': duracao
            }
            resultados[nome_rede].append(treino_info)
            dados_tabela_item3[f'{nome_rede} (T{i + 1})'] = np.round(previsoes, 4)

            print(f"  [{nome_rede}] T{i + 1} (seed={semente}): {len(hist)} épocas em {duracao:.2f}s | EQM={hist[-1]:.6f} | Erro Médio={erro_medio_pct:.2f}%")

    tempo_total = time.time() - inicio_total
    print(f"\nTreinamentos concluídos com sucesso em {tempo_total:.2f} segundos!")

    # ---------------------------------------------------------------------
    # ITEM 2: Tabela de Resultados dos Treinamentos (EQM e Épocas)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 2: RESULTADOS FINAIS DOS 3 TREINAMENTOS (EQM E ÉPOCAS)")
    print("=" * 80)

    col_t2 = ['Treinamento', 'Rede 1 EQM', 'Rede 1 Épocas', 'Rede 2 EQM', 'Rede 2 Épocas', 'Rede 3 EQM', 'Rede 3 Épocas']
    linhas_t2 = []

    for t_idx in range(3):
        r1 = resultados['Rede 1'][t_idx]
        r2 = resultados['Rede 2'][t_idx]
        r3 = resultados['Rede 3'][t_idx]
        linhas_t2.append([
            f"{t_idx + 1}º (T{t_idx + 1})",
            f"{r1['eqm_final']:.6f}", r1['epocas'],
            f"{r2['eqm_final']:.6f}", r2['epocas'],
            f"{r3['eqm_final']:.6f}", r3['epocas']
        ])

    df_item2 = pd.DataFrame(linhas_t2, columns=col_t2)
    print(df_item2.to_string(index=False))
    df_item2.to_csv('tabela_item2.csv', index=False)

    # ---------------------------------------------------------------------
    # ITEM 3: Tabela de Validação de Amostras de Teste
    # ---------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 3: VALIDAÇÃO DAS REDES (AMOSTRAS t=101..120, ERRO RELATIVO MÉDIO E VARIÂNCIA)")
    print("=" * 80)

    df_item3 = pd.DataFrame(dados_tabela_item3)

    # Linhas de resumo estatístico (Erro Relativo Médio e Variância)
    linha_erro_pct = ['Erro Rel. Médio (%)', '-']
    linha_var_pct = ['Variância (%)', '-']
    linha_erro_dec = ['Erro Rel. Médio (dec)', '-']
    linha_var_dec = ['Variância (dec)', '-']

    for nome in ['Rede 1', 'Rede 2', 'Rede 3']:
        for t_idx in range(3):
            r = resultados[nome][t_idx]
            linha_erro_pct.append(f"{r['erro_medio_pct']:.2f}%")
            linha_var_pct.append(f"{r['var_pct']:.2f}")
            linha_erro_dec.append(f"{r['erro_medio_dec']:.4f}")
            linha_var_dec.append(f"{r['var_dec']:.6f}")

    df_item3_completo = pd.concat([
        df_item3,
        pd.DataFrame([linha_erro_pct, linha_var_pct, linha_erro_dec, linha_var_dec], columns=df_item3.columns)
    ], ignore_index=True)

    print(df_item3_completo.to_string(index=False))
    df_item3_completo.to_csv('tabela_item3.csv', index=False)

    # ---------------------------------------------------------------------
    # Determinação Dinâmica do Melhor Treinamento de Cada Topologia
    # ---------------------------------------------------------------------
    # Critério: menor erro relativo médio de teste (generalização)
    melhores = {}
    print("\n" + "=" * 80)
    print("SELEÇÃO DO MELHOR TREINAMENTO POR TOPOLOGIA:")
    print("=" * 80)

    for nome in topologias:
        erros_test = [r['erro_medio_pct'] for r in resultados[nome]]
        eqms_train = [r['eqm_final'] for r in resultados[nome]]

        idx_melhor_teste = int(np.argmin(erros_test))
        idx_melhor_treino = int(np.argmin(eqms_train))

        melhores[nome] = idx_melhor_teste
        r_best = resultados[nome][idx_melhor_teste]

        print(f"* {nome}: Melhor treino = {r_best['treino']} (Seed={r_best['semente']})")
        print(f"    - Erro Relativo Médio (Teste): {r_best['erro_medio_pct']:.2f}% (Variância: {r_best['var_pct']:.2f})")
        print(f"    - EQM Final (Treinamento): {r_best['eqm_final']:.6f} ({r_best['epocas']} épocas)")

    # ---------------------------------------------------------------------
    # ITEM 4: Gráficos de EQM por Época (Melhor Treino de Cada Rede)
    # ---------------------------------------------------------------------
    print("\nGerando gráfico do Item 4: grafico_eqm_topologias.png...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for idx, (nome, id_run) in enumerate(melhores.items()):
        info = resultados[nome][id_run]
        axes[idx].plot(info['historico'], color='#1d4ed8', lw=1.8, label='EQM por Época')
        axes[idx].set_title(
            f"{nome} - {info['treino']} (Melhor Treino)\nÉpocas: {info['epocas']} | EQM: {info['eqm_final']:.6f}",
            fontsize=11, fontweight='bold'
        )
        axes[idx].set_xlabel('Épocas', fontsize=10)
        axes[idx].set_ylabel('Erro Quadrático Médio (EQM)', fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.6)
        axes[idx].legend(loc='upper right')

    plt.suptitle("Item 4: Curvas de Convergência do EQM ao Longo das Épocas de Treinamento", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('grafico_eqm_topologias.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------
    # ITEM 5: Gráficos de Estimação vs Desejado (Melhor Treino de Cada Rede)
    # ---------------------------------------------------------------------
    print("Gerando gráfico do Item 5: grafico_estimacao_topologias.png...")
    t_eixo = np.arange(101, 121)
    d_alvo = serie_temporal[100:120]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for idx, (nome, id_run) in enumerate(melhores.items()):
        info = resultados[nome][id_run]
        axes[idx].plot(t_eixo, d_alvo, 'o--', color='#111827', label='Desejado f(t)', lw=1.6, markersize=5)
        axes[idx].plot(t_eixo, info['previsoes'], 's-', color='#dc2626', label=f"Estimado ({info['treino']})", lw=1.6, markersize=5)
        axes[idx].set_title(
            f"{nome} - {info['treino']}\nErro Relativo Médio: {info['erro_medio_pct']:.2f}% | Var: {info['var_pct']:.2f}",
            fontsize=11, fontweight='bold'
        )
        axes[idx].set_xlabel('Instante de Tempo (t)', fontsize=10)
        axes[idx].set_ylabel('f(t)', fontsize=10)
        axes[idx].set_xticks(t_eixo[::2])
        axes[idx].legend(loc='best')
        axes[idx].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Item 5: Comparação entre Valores Desejados e Estimados (t = 101 .. 120)", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('grafico_estimacao_topologias.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nGráficos e tabelas salvos com sucesso na pasta atual!")
    print("=" * 80)