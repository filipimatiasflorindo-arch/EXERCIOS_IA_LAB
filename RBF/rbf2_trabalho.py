

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kmeans_custom(X, k, max_iter=500, tol=1e-10):
    """
    Estágio 1 (Não-supervisionado): Algoritmo K-Means para determinação dos
    centros dos clusters e suas respectivas variâncias.
    Inicializa os k centros com as primeiras k amostras do conjunto de treinamento.
    """
    centers = X[:k].copy().astype(float)

    for _ in range(1, max_iter + 1):
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(np.abs(new_centers - centers) <= tol):
            centers = new_centers
            break
        centers = new_centers

    variances = np.array([
        np.mean(np.sum((X[labels == i] - centers[i]) ** 2, axis=1))
        for i in range(k)
    ])

    return centers, variances, labels


def calcular_matriz_rbf(X, centers, variances):
    """
    Calcula as saídas gaussianas da camada escondida:
    phi_j(x) = exp(- ||x - c_j||^2 / (2 * sigma_j^2))
    """
    k = len(centers)
    phi = np.column_stack([
        np.exp(-np.sum((X - centers[j]) ** 2, axis=1) / (2.0 * variances[j]))
        for j in range(k)
    ])
    return phi


def treinar_camada_saida(Z, d, eta=0.01, eps=1e-7, max_epochs=100000, seed=1):
    """
    Estágio 2 (Supervisionado): Regra Delta Generalizada para camada de saída linear.
    Inicializa os pesos aleatoriamente no intervalo [0, 1] com semente distinta por treino.
    """
    rng = np.random.default_rng(seed)
    # Também compatível com np.random.seed(seed) para reprodutibilidade clássica
    np.random.seed(seed)
    P, num_inputs = Z.shape
    W = np.random.uniform(0.0, 1.0, size=num_inputs)

    prev_eqm = float('inf')
    historico_eqm = []

    for epoch in range(1, max_epochs + 1):
        for p in range(P):
            y_p = np.dot(W, Z[p])
            erro_p = d[p] - y_p
            W += eta * erro_p * Z[p]

        y_all = np.dot(Z, W)
        eqm = np.mean((d - y_all) ** 2)
        historico_eqm.append(eqm)

        if abs(prev_eqm - eqm) <= eps:
            break
        prev_eqm = eqm

    return W, epoch, eqm, historico_eqm


def main():
    print("=" * 90)
    print("CEFET-MG - CAMPUS VIII VARGINHA - LAB. INTELIGÊNCIA ARTIFICIAL")
    print("RESOLUÇÃO DO TRABALHO: RBF 2 (APROXIMAÇÃO FUNCIONAL - INJEÇÃO ELETRÔNICA)")
    print("=" * 90)

    # Carregamento dos conjuntos de treinamento (150 amostras) e teste (15 amostras)
    df_treino = pd.read_csv('treinamento_2.csv')
    df_teste = pd.read_csv('teste_2.csv')

    X_train = df_treino[['x1', 'x2', 'x3']].values
    d_train = df_treino['d'].values

    X_test = df_teste[['x1', 'x2', 'x3']].values
    d_test = df_teste['d'].values

    print(f"\n[+] Conjuntos carregados:")
    print(f"    - Treinamento: {len(X_train)} padrões (x1, x2, x3 -> d)")
    print(f"    - Validação/Teste: {len(X_test)} padrões (x1, x2, x3 -> d)")

    topologias = [
        ('Rede 1', 5),
        ('Rede 2', 10),
        ('Rede 3', 15),
    ]
    sementes = [1, 2, 3]  # 3 treinamentos com gerador reiniciado (T1, T2, T3)
    eta = 0.01
    eps = 1e-7
    x0 = -1.0  # Entrada do limiar (bias) na convenção w0 = theta

    resultados = {}

    # -------------------------------------------------------------------------
    # ATIVIDADE 1: Execução dos 3 treinamentos para cada topologia (N1 = 5, 10, 15)
    # -------------------------------------------------------------------------
    for nome_rede, n1 in topologias:
        centers, variances, labels = kmeans_custom(X_train, k=n1)

        phi_train = calcular_matriz_rbf(X_train, centers, variances)
        phi_test = calcular_matriz_rbf(X_test, centers, variances)

        Z_train = np.column_stack([np.full(len(X_train), x0), phi_train])
        Z_test = np.column_stack([np.full(len(X_test), x0), phi_test])

        resultados[nome_rede] = {
            'N1': n1,
            'centers': centers,
            'variances': variances,
            'treinos': []
        }

        for idx_t, seed in enumerate(sementes, start=1):
            W, epocas, eqm_final, hist_eqm = treinar_camada_saida(
                Z_train, d_train, eta=eta, eps=eps, seed=seed
            )

            # Predição no conjunto de teste
            y_test = np.dot(Z_test, W)

            # Erro relativo percentual em cada padrão de teste: |d - y| / |d| * 100
            erros_relativos = (np.abs(d_test - y_test) / np.abs(d_test)) * 100.0
            erm = np.mean(erros_relativos)
            var_pop = np.var(erros_relativos, ddof=0)
            var_amostral = np.var(erros_relativos, ddof=1)

            resultados[nome_rede]['treinos'].append({
                'nome': f'T{idx_t}',
                'seed': seed,
                'W': W,
                'epocas': epocas,
                'eqm': eqm_final,
                'historico_eqm': hist_eqm,
                'y_test': y_test,
                'erm': erm,
                'var_pop': var_pop,
                'var_amostral': var_amostral,
            })

    # -------------------------------------------------------------------------
    # ATIVIDADE 2: Tabela de EQM e Épocas dos 3 treinamentos por topologia
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ATIVIDADE 2: RESULTADOS FINAIS DOS TREINAMENTOS (EQM E ÉPOCAS)")
    print("=" * 90)

    linhas_ativ2 = []
    for i, rotulo in enumerate(['1º (T1)', '2º (T2)', '3º (T3)']):
        r1 = resultados['Rede 1']['treinos'][i]
        r2 = resultados['Rede 2']['treinos'][i]
        r3 = resultados['Rede 3']['treinos'][i]
        linhas_ativ2.append({
            'Treinamento': rotulo,
            'Rede 1 - EQM': f"{r1['eqm']:.6f}",
            'Rede 1 - Épocas': r1['epocas'],
            'Rede 2 - EQM': f"{r2['eqm']:.6f}",
            'Rede 2 - Épocas': r2['epocas'],
            'Rede 3 - EQM': f"{r3['eqm']:.6f}",
            'Rede 3 - Épocas': r3['epocas'],
        })

    df_ativ2 = pd.DataFrame(linhas_ativ2)
    print(df_ativ2.to_string(index=False))

    # -------------------------------------------------------------------------
    # ATIVIDADE 3: Validação no Conjunto de Teste (Erro Relativo Médio e Variância)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ATIVIDADE 3: VALIDAÇÃO DAS REDES NO CONJUNTO DE TESTE")
    print("=" * 90)

    df_val = df_teste.copy()
    metricas_resumo = []

    for nome_rede, _ in topologias:
        for tr in resultados[nome_rede]['treinos']:
            col_name = f"{nome_rede}_{tr['nome']}"
            df_val[col_name] = np.round(tr['y_test'], 4)
            metricas_resumo.append({
                'Topologia': nome_rede,
                'Treino': tr['nome'],
                'EQM Treino': tr['eqm'],
                'Épocas': tr['epocas'],
                'ERM (%)': tr['erm'],
                'Variância Pop. (%)': tr['var_pop'],
                'Variância Amostral (%)': tr['var_amostral'],
            })

    print(df_val.to_string(index=False))
    print("-" * 90)
    df_metricas = pd.DataFrame(metricas_resumo)
    print("\nResumo de Erro Relativo Médio (%) e Variância (%) no Teste:")
    df_metricas_fmt = df_metricas.copy()
    df_metricas_fmt['EQM Treino'] = df_metricas_fmt['EQM Treino'].map(lambda v: f"{v:.8f}")
    df_metricas_fmt['ERM (%)'] = df_metricas_fmt['ERM (%)'].map(lambda v: f"{v:.4f}%")
    df_metricas_fmt['Variância Pop. (%)'] = df_metricas_fmt['Variância Pop. (%)'].map(lambda v: f"{v:.4f}%")
    df_metricas_fmt['Variância Amostral (%)'] = df_metricas_fmt['Variância Amostral (%)'].map(lambda v: f"{v:.4f}%")
    print(df_metricas_fmt.to_string(index=False))

    df_val.to_csv('resultado_validacao_teste2.csv', index=False)
    print("\n[+] Tabela de validação salva em 'resultado_validacao_teste2.csv'")

    # -------------------------------------------------------------------------
    # ATIVIDADE 4: Gráficos de EQM x Época (não superpostos numa mesma folha)
    # Seleção do melhor treinamento {T1, T2 ou T3} em cada topologia
    # Critério: menor EQM de treinamento (desempate por menor ERM de teste / épocas)
    # -------------------------------------------------------------------------
    melhores = {}
    for nome_rede, _ in topologias:
        treinos = resultados[nome_rede]['treinos']
        # Seleciona o treinamento com menor ERM no conjunto de validação
        melhor_tr = min(treinos, key=lambda t: (round(t['eqm'], 6), t['erm'], t['epocas']))
        melhores[nome_rede] = melhor_tr

    fig, axes = plt.subplots(3, 1, figsize=(10, 13))
    cores = ['#1f4e79', '#2e75b6', '#0070c0']

    for idx, (nome_rede, n1) in enumerate(topologias):
        tr = melhores[nome_rede]
        hist = tr['historico_eqm']
        epocas_x = np.arange(1, len(hist) + 1)

        ax = axes[idx]
        ax.plot(epocas_x, hist, color=cores[idx], linewidth=2.0)
        ax.set_title(
            f"{nome_rede} (N1 = {n1:02d}) — Melhor Treinamento: {tr['nome']} | "
            f"Épocas: {tr['epocas']} | EQM Final: {tr['eqm']:.6f} | ERM Teste: {tr['erm']:.2f}%",
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel("Época de Treinamento", fontsize=10)
        ax.set_ylabel("Erro Quadrático Médio (EQM)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(
        "Evolução do Erro Quadrático Médio (EQM) por Época — Melhores Treinamentos por Topologia",
        fontsize=13,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    plt.savefig('graficos_rbf2.png', dpi=300)
    plt.close()

    print("\n" + "=" * 90)
    print("ATIVIDADE 4: GRÁFICOS DE CONVERGÊNCIA DO EQM")
    print("=" * 90)
    for nome_rede, _ in topologias:
        tr = melhores[nome_rede]
        print(f"  - {nome_rede} (N1={resultados[nome_rede]['N1']:02d}): Melhor treino = {tr['nome']} "
              f"(Épocas = {tr['epocas']}, EQM = {tr['eqm']:.6f}, ERM = {tr['erm']:.4f}%)")
    print("[+] Três gráficos impressos numa mesma folha (não superpostos) salvos em 'graficos_rbf2.png'")

    # -------------------------------------------------------------------------
    # ATIVIDADE 5: Indicação da Topologia e Configuração Mais Adequada
    # -------------------------------------------------------------------------
    melhor_global_rede = min(topologias, key=lambda item: melhores[item[0]]['erm'])[0]
    melhor_global_tr = melhores[melhor_global_rede]

    print("\n" + "=" * 90)
    print("ATIVIDADE 5: SELEÇÃO DA ARQUITETURA E TREINAMENTO MAIS ADEQUADOS")
    print("=" * 90)
    print(f"Topologia Recomendada: {melhor_global_rede} (RBF com N1 = {resultados[melhor_global_rede]['N1']} neurônios ocultos)")
    print(f"Configuração Final Selecionada: Treinamento {melhor_global_tr['nome']}")
    print(f"  - EQM de Treinamento: {melhor_global_tr['eqm']:.6f}")
    print(f"  - Número de Épocas: {melhor_global_tr['epocas']}")
    print(f"  - Erro Relativo Médio no Teste: {melhor_global_tr['erm']:.4f}%")
    print(f"  - Variância no Teste: {melhor_global_tr['var_pop']:.4f}% (amostral: {melhor_global_tr['var_amostral']:.4f}%)")
    print("=" * 90)


if __name__ == '__main__':
    main()

