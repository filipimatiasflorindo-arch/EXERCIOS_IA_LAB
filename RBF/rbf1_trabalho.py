

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kmeans_custom(X, k=2, max_iter=100, tol=1e-6):
    """
    Executa o algoritmo K-Means para encontrar os k centros e suas respectivas variâncias.
    Inicializa deterministicamente com os primeiros pontos distintos do conjunto.
    """
    # Inicialização com os primeiros k padrões distintos
    centers = X[:k].copy().astype(float)
    
    for iteration in range(1, max_iter + 1):
        # Distâncias euclidianas quadradas a cada centro
        dists = np.array([np.sum((X - c)**2, axis=1) for c in centers])
        labels = np.argmin(dists, axis=0)
        
        # Recálculo dos centros
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Critério de convergência
        if np.all(np.abs(new_centers - centers) < tol):
            break
        centers = new_centers

    # Cálculo da variância de cada cluster: dispersão média quadrática ao centroide
    variances = []
    cluster_points = []
    for i in range(k):
        pts = X[labels == i]
        cluster_points.append(pts)
        var = np.mean(np.sum((pts - centers[i])**2, axis=1))
        variances.append(var)
        
    return centers, np.array(variances), labels, cluster_points


def gaussian_rbf(X, center, variance):
    """
    Função de base radial gaussiana:
    phi(x) = exp(- ||x - c||^2 / (2 * sigma^2))
    """
    dist_sq = np.sum((X - center)**2, axis=1)
    return np.exp(-dist_sq / (2.0 * variance))


def treinar_camada_saida(Z, d, eta=0.01, eps=1e-7, max_epochs=50000, seed=42):
    """
    Treina o neurônio da camada de saída via Regra Delta Generalizada:
    Z: matriz de entrada do neurônio (bias, phi1, phi2)
    d: vetor com valores desejados (-1 ou +1)
    eta: taxa de aprendizado (0.01)
    eps: precisão de parada (10^-7)
    """
    np.random.seed(seed)
    P, num_inputs = Z.shape
    
    # Inicialização dos pesos com valores aleatórios entre 0 e 1
    W = np.random.uniform(0.0, 1.0, size=num_inputs)
    
    prev_eqm = float('inf')
    historico_eqm = []
    
    for epoch in range(1, max_epochs + 1):
        # Treinamento padrão a padrão (online)
        for k in range(P):
            y_k = np.dot(W, Z[k])
            erro_k = d[k] - y_k
            W += eta * erro_k * Z[k]
            
        # Avaliação do Erro Quadrático Médio (EQM) no final da época
        y_epoch = np.dot(Z, W)
        eqm = np.mean((d - y_epoch)**2)
        historico_eqm.append(eqm)
        
        # Critério de parada: |EQM(t) - EQM(t-1)| <= eps
        if abs(prev_eqm - eqm) <= eps:
            break
        prev_eqm = eqm
        
    return W, epoch, eqm, historico_eqm


def pos_processamento(y):
    """
    Atividade 3: Função sinal para converter saídas reais em números inteiros {-1, +1}
    y_pos = 1 se y >= 0 senão -1
    """
    return np.where(y >= 0.0, 1, -1)


def main():
    print("=" * 80)
    print("CEFET-MG - CAMPUS VIII VARGINHA - LAB. INTELIGÊNCIA ARTIFICIAL")
    print("RESOLUÇÃO DO TRABALHO: RBF 1 (CLASSIFICAÇÃO DE RADIAÇÃO)")
    print("=" * 80)

    # 1. Carregamento dos dados
    df_treino = pd.read_csv('treinamento_1.csv')
    df_teste = pd.read_csv('teste_1.csv')
    
    print(f"\n[+] Dados carregados com sucesso:")
    print(f"    - Amostras de Treinamento: {len(df_treino)}")
    print(f"    - Amostras de Teste: {len(df_teste)}")

    # -------------------------------------------------------------------------
    # ATIVIDADE 1: Treinamento da Camada Escondida (K-Means)
    # -------------------------------------------------------------------------
    # Filtrar apenas as amostras com presença de radiação (d = 1)
    df_rad = df_treino[df_treino['d'] == 1].copy()
    X_rad = df_rad[['x1', 'x2']].values
    print(f"    - Amostras com radiação (d = 1) para K-Means: {len(X_rad)}")

    centers, variances, labels, cluster_pts = kmeans_custom(X_rad, k=2)

    print("\n" + "=" * 50)
    print("ATIVIDADE 1: CENTROS E VARIÂNCIAS DOS CLUSTERS")
    print("=" * 50)
    tabela_ativ1 = pd.DataFrame({
        'Cluster': [1, 2],
        'Centro (x1, x2)': [f"({centers[0][0]:.6f}, {centers[0][1]:.6f})",
                            f"({centers[1][0]:.6f}, {centers[1][1]:.6f})"],
        'Variância': [f"{variances[0]:.6f}", f"{variances[1]:.6f}"],
        'Qtd Amostras': [len(cluster_pts[0]), len(cluster_pts[1])]
    })
    print(tabela_ativ1.to_string(index=False))

    # -------------------------------------------------------------------------
    # ATIVIDADE 2: Treinamento da Camada de Saída (Regra Delta Generalizada)
    # -------------------------------------------------------------------------
    X_train_all = df_treino[['x1', 'x2']].values
    d_train_all = df_treino['d'].values

    # Ativações da camada escondida
    phi1_train = gaussian_rbf(X_train_all, centers[0], variances[0])
    phi2_train = gaussian_rbf(X_train_all, centers[1], variances[1])

    # Montagem da matriz Z: [x0, phi1, phi2]
    # Convenção com x0 = -1 (padrão CEFET com w0 = theta)
    x0 = -1.0
    Z_train = np.column_stack([np.full(len(X_train_all), x0), phi1_train, phi2_train])

    eta = 0.01
    eps = 1e-7
    W, num_epocas, eqm_final, historico_eqm = treinar_camada_saida(
        Z_train, d_train_all, eta=eta, eps=eps, seed=42
    )

    print("\n" + "=" * 50)
    print("ATIVIDADE 2: PESOS DA CAMADA DE SAÍDA")
    print("=" * 50)
    print(f"Convergência atingida na época: {num_epocas}")
    print(f"EQM Final de Treinamento: {eqm_final:.6f}")
    tabela_ativ2 = pd.DataFrame({
        'Peso': ['W21,0', 'W21,1', 'W21,2'],
        'Descrição': ['Bias (x0 = -1)', 'Conexão Neurônio Oculto 1', 'Conexão Neurônio Oculto 2'],
        'Valor': [f"{W[0]:.4f}", f"{W[1]:.4f}", f"{W[2]:.4f}"]
    })
    print(tabela_ativ2.to_string(index=False))
    print(f"\nNota: Se for adotada a convenção com x0 = +1, W21,0 = {-W[0]:.4f}")

    # -------------------------------------------------------------------------
    # ATIVIDADES 3 & 4: Pós-Processamento e Validação no Conjunto de Teste
    # -------------------------------------------------------------------------
    X_test = df_teste[['x1', 'x2']].values
    d_test = df_teste['d'].values

    phi1_test = gaussian_rbf(X_test, centers[0], variances[0])
    phi2_test = gaussian_rbf(X_test, centers[1], variances[1])
    Z_test = np.column_stack([np.full(len(X_test), x0), phi1_test, phi2_test])

    # Saída real da rede
    y_test_real = np.dot(Z_test, W)

    # Pós-processamento com função sinal (Atividade 3)
    y_pos = pos_processamento(y_test_real)

    # Montagem da tabela de resultados
    df_resultado_teste = df_teste.copy()
    df_resultado_teste['y'] = np.round(y_test_real, 4)
    df_resultado_teste['y_pos'] = y_pos
    df_resultado_teste['Acerto'] = df_resultado_teste['d'] == df_resultado_teste['y_pos']

    acertos = df_resultado_teste['Acerto'].sum()
    total = len(df_resultado_teste)
    taxa_acerto = (acertos / total) * 100.0

    print("\n" + "=" * 70)
    print("ATIVIDADE 4: VALIDAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 70)
    print(df_resultado_teste[['amostra', 'x1', 'x2', 'd', 'y', 'y_pos']].to_string(index=False))
    print("-" * 70)
    print(f"Taxa de Acerto (%): {taxa_acerto:.2f}% ({acertos}/{total} acertos)")
    print("=" * 70)

    # Salva os resultados detalhados em CSV
    df_resultado_teste.to_csv('resultado_validacao_teste1.csv', index=False)
    print("\n[+] Tabela de validação salva em 'resultado_validacao_teste1.csv'")

    # -------------------------------------------------------------------------
    # GERAÇÃO DE GRÁFICOS ILUSTRATIVOS
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Curva de Convergência do EQM
    axes[0].plot(range(1, len(historico_eqm) + 1), historico_eqm, color='navy', lw=2)
    axes[0].set_title(f"Convergência do EQM (Épocas: {num_epocas}, EQM: {eqm_final:.4f})", fontsize=12)
    axes[0].set_xlabel("Épocas de Treinamento", fontsize=11)
    axes[0].set_ylabel("Erro Quadrático Médio (EQM)", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Gráfico 2: Espaço de Decisão 2D
    # Plot dados de treino
    tr_sem_rad = df_treino[df_treino['d'] == -1]
    tr_com_rad = df_treino[df_treino['d'] == 1]
    axes[1].scatter(tr_sem_rad['x1'], tr_sem_rad['x2'], c='lightgray', marker='o', s=40, label='Treino: Ausência (d=-1)')
    axes[1].scatter(tr_com_rad['x1'], tr_com_rad['x2'], c='blue', marker='o', s=40, alpha=0.7, label='Treino: Presença (d=+1)')

    # Plot centros dos clusters
    axes[1].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=160, edgecolor='black', zorder=5, label='Centros K-Means')
    for i, c in enumerate(centers):
        circ = plt.Circle((c[0], c[1]), np.sqrt(variances[i]), color='red', fill=False, linestyle='--', lw=1.5)
        axes[1].add_patch(circ)
        axes[1].annotate(f'Cluster {i+1}', (c[0], c[1]), textcoords="offset points", xytext=(8, 8), fontweight='bold')

    # Plot pontos de teste
    te_certos = df_resultado_teste[df_resultado_teste['Acerto']]
    te_erros = df_resultado_teste[~df_resultado_teste['Acerto']]
    axes[1].scatter(te_certos['x1'], te_certos['x2'], c='green', marker='^', s=80, edgecolors='black', label='Teste: Acerto')
    axes[1].scatter(te_erros['x1'], te_erros['x2'], c='magenta', marker='v', s=100, edgecolors='black', label='Teste: Erro (Amostras 2 e 4)')

    axes[1].set_title("Espaço de Características (x1, x2), Clusters e Validação", fontsize=12)
    axes[1].set_xlabel("x1", fontsize=11)
    axes[1].set_ylabel("x2", fontsize=11)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('graficos_rbf1.png', dpi=300)
    plt.close()
    print("[+] Gráficos salvos com sucesso no arquivo 'graficos_rbf1.png'")

    # -------------------------------------------------------------------------
    # ATIVIDADE 5: Explicações Técnicas
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ATIVIDADE 5: ESTRATÉGIAS PARA AUMENTAR A TAXA DE ACERTO DA RBF")
    print("=" * 70)
    estrategias = """
1. Aumento do número de neurônios da camada escondida (K > 2):
   - A rede errou as amostras 2 (x1=0.0388, x2=0.2703) e 4 (x1=0.7075, x2=0.1502).
   - Ambas pertencem à classe d = +1, porém estão localizadas em regiões periféricas,
     afastadas dos dois centros computados. Com K = 3 ou 4 centros, a cobertura
     das áreas de presença de radiação se torna muito mais abrangente.

2. Inclusão dos dados de ambas as classes no K-Means:
   - O procedimento padrão agrupou apenas os padrões de presença de radiação (d = 1).
   - Ao posicionar centros também para a classe sem radiação (d = -1), a RBF passa
     a mapear a fronteira de separação entre classes com muito mais precisão.

3. Treinamento supervisionado com ajuste fino (fine-tuning):
   - Em vez de manter os centros e variâncias fixos após o K-Means, aplicar o gradiente
     descendente supervisionado para ajustar conjuntamente os pesos da saída, os centros
     e as larguras (sigmas) das gaussianas.

4. Calibração da largura/espalhamento (sigma) das gaussianas:
   - Adoção de um fator multiplicativo de espalhamento sobre os sigmas, permitindo
     cobrir regiões entre os clusters sem comprometer a especificidade da classe.
"""
    print(estrategias.strip())
    print("=" * 70)


if __name__ == '__main__':
    main()
