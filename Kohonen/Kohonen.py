import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DADOS DO APÊNDICE (120 Amostras de Treinamento) E DE TESTE (12)
# =====================================================================

dados_treinamento = np.array([
    # Classe A: Amostras 1 a 20
    [0.2417, 0.2857, 0.2397], [0.2268, 0.2874, 0.2153], [0.1975, 0.3315, 0.1965],
    [0.3414, 0.3166, 0.1074], [0.2587, 0.1918, 0.2634], [0.2455, 0.2075, 0.1344],
    [0.3163, 0.1679, 0.1725], [0.2704, 0.2605, 0.1411], [0.1871, 0.2965, 0.1231],
    [0.3474, 0.2715, 0.1958], [0.2059, 0.2928, 0.2839], [0.2442, 0.2272, 0.2384],
    [0.2126, 0.3437, 0.1128], [0.2562, 0.2542, 0.1599], [0.1640, 0.2289, 0.2627],
    [0.2795, 0.1880, 0.1627], [0.3463, 0.1513, 0.2281], [0.3430, 0.1508, 0.1881],
    [0.1981, 0.2821, 0.1294], [0.2322, 0.3025, 0.2191],
    # Classe B: Amostras 21 a 60
    [0.7352, 0.2722, 0.6962], [0.7191, 0.1825, 0.7470], [0.6921, 0.1537, 0.8172],
    [0.6833, 0.2048, 0.8490], [0.8012, 0.2684, 0.7673], [0.7860, 0.1734, 0.7198],
    [0.7205, 0.1542, 0.7295], [0.6549, 0.3288, 0.8153], [0.6968, 0.3173, 0.7389],
    [0.7448, 0.2095, 0.6847], [0.6746, 0.3277, 0.6725], [0.7897, 0.2801, 0.7679],
    [0.8399, 0.3067, 0.7003], [0.8065, 0.3206, 0.7205], [0.8357, 0.3220, 0.7879],
    [0.7438, 0.3230, 0.8384], [0.8172, 0.3319, 0.7628], [0.8248, 0.2614, 0.8405],
    [0.6979, 0.2142, 0.7309], [0.6804, 0.3181, 0.7017], [0.6973, 0.3194, 0.7522],
    [0.7910, 0.2239, 0.7018], [0.7052, 0.2148, 0.6866], [0.8088, 0.1908, 0.7563],
    [0.7640, 0.1676, 0.6994], [0.7616, 0.2881, 0.8087], [0.8188, 0.2461, 0.7273],
    [0.7920, 0.3178, 0.7497], [0.7802, 0.1871, 0.8102], [0.7332, 0.2543, 0.8194],
    [0.6921, 0.1529, 0.7759], [0.6833, 0.2197, 0.6943], [0.7860, 0.1745, 0.7639],
    [0.8009, 0.3082, 0.8491], [0.7793, 0.1935, 0.6738], [0.7373, 0.2698, 0.7864],
    [0.7048, 0.2380, 0.7825], [0.8393, 0.2857, 0.7733], [0.6878, 0.2126, 0.6961],
    [0.6651, 0.3492, 0.6737],
    # Classe C: Amostras 61 a 120
    [0.4856, 0.6600, 0.4798], [0.4114, 0.7220, 0.5106], [0.5671, 0.7935, 0.5929],
    [0.4875, 0.7928, 0.5532], [0.5172, 0.7147, 0.5774], [0.5483, 0.6773, 0.4842],
    [0.5740, 0.6682, 0.5335], [0.4587, 0.6981, 0.5900], [0.5794, 0.7410, 0.4759],
    [0.4712, 0.6734, 0.5677], [0.5126, 0.8141, 0.5224], [0.5557, 0.7749, 0.4342],
    [0.4916, 0.8267, 0.4586], [0.4629, 0.8129, 0.4950], [0.5850, 0.7358, 0.5107],
    [0.4435, 0.7030, 0.4594], [0.4155, 0.7516, 0.5524], [0.4887, 0.7027, 0.5886],
    [0.5462, 0.7378, 0.5107], [0.5251, 0.8124, 0.5686], [0.4635, 0.7339, 0.5638],
    [0.5907, 0.7144, 0.4718], [0.4982, 0.8335, 0.4597], [0.5242, 0.7325, 0.4079],
    [0.4075, 0.8372, 0.4271], [0.5934, 0.8284, 0.5107], [0.5463, 0.6766, 0.5639],
    [0.4403, 0.8495, 0.4806], [0.4531, 0.7760, 0.5276], [0.5109, 0.7387, 0.5373],
    [0.5383, 0.7780, 0.4955], [0.5679, 0.7156, 0.5022], [0.5762, 0.7781, 0.5908],
    [0.5997, 0.7504, 0.5678], [0.4138, 0.6975, 0.5148], [0.5490, 0.6674, 0.4472],
    [0.4719, 0.7527, 0.4401], [0.4458, 0.8063, 0.4253], [0.4983, 0.8131, 0.5625],
    [0.5742, 0.6789, 0.5997], [0.5289, 0.7354, 0.4718], [0.5927, 0.7738, 0.5390],
    [0.5199, 0.7131, 0.4028], [0.5716, 0.6558, 0.4451], [0.5075, 0.7045, 0.4233],
    [0.4886, 0.7004, 0.4608], [0.5527, 0.8243, 0.5772], [0.4816, 0.6969, 0.4678],
    [0.5809, 0.6557, 0.4266], [0.5881, 0.7565, 0.4003], [0.5334, 0.8446, 0.4934],
    [0.4603, 0.7992, 0.4816], [0.5491, 0.6504, 0.4063], [0.4288, 0.8455, 0.5047],
    [0.5636, 0.7884, 0.5417], [0.5349, 0.6736, 0.4541], [0.5569, 0.8393, 0.5652],
    [0.4729, 0.7702, 0.5325], [0.5472, 0.8454, 0.5449], [0.5805, 0.7349, 0.4464]
])

amostras_teste = np.array([
    [0.2471, 0.1778, 0.2905],
    [0.8240, 0.2223, 0.7041],
    [0.4960, 0.7231, 0.5866],
    [0.2923, 0.2041, 0.2234],
    [0.8118, 0.2668, 0.7484],
    [0.4837, 0.8200, 0.4792],
    [0.3248, 0.2629, 0.2375],
    [0.7209, 0.2116, 0.7821],
    [0.5259, 0.6522, 0.5957],
    [0.2075, 0.1669, 0.1745],
    [0.7830, 0.3171, 0.7888],
    [0.5393, 0.7510, 0.5682]
])

# Rótulos reais das amostras de treinamento conforme enunciado:
# 1..20 -> A | 21..60 -> B | 61..120 -> C
rotulos_treinamento = (['A'] * 20) + (['B'] * 40) + (['C'] * 60)

# =====================================================================
# 2. IMPLEMENTAÇÃO DA REDE DE KOHONEN (SOM 4x4)
# =====================================================================

class RedeKohonen:
    def __init__(self, linhas=4, colunas=4, dim_entrada=3, eta=0.001, raio=1):
        self.linhas = linhas
        self.colunas = colunas
        self.num_neuronios = linhas * colunas # 16 neurônios (1 a 16)
        self.dim_entrada = dim_entrada
        self.eta = eta
        self.raio = raio
        
        # Inicializa pesos aleatoriamente no intervalo dos dados [0.1, 0.85]
        np.random.seed(42)
        self.W = np.random.uniform(0.15, 0.85, (self.num_neuronios, dim_entrada))
        
        # Coordenadas (linha, coluna) de cada neurônio (0 a 15 -> 1 a 16 no diagrama)
        self.coordenadas = np.array([(i // colunas, i % colunas) for i in range(self.num_neuronios)])

    def obter_vizinhos(self, idx_vencedor):
        """
        Retorna os índices do neurônio vencedor e de seus vizinhos com raio = 1.
        Pelo diagrama esquemático (conexões horizontais e verticais diretas),
        utiliza-se a distância de Manhattan <= raio (vizinhança de Von Neumann).
        """
        coord_vencedor = self.coordenadas[idx_vencedor]
        distancias_grid = np.sum(np.abs(self.coordenadas - coord_vencedor), axis=1)
        return np.where(distancias_grid <= self.raio)[0]

    def encontrar_vencedor(self, x):
        """Encontra o neurônio vencedor (BMU) pela menor distância Euclidiana."""
        distancias = np.linalg.norm(self.W - x, axis=1)
        return np.argmin(distancias)

    def treinar(self, X, epocas=1500):
        """Treina a rede atualizando o vencedor e sua vizinhança de raio 1."""
        for _ in range(epocas):
            indices = np.random.permutation(len(X))
            for idx in indices:
                x = X[idx]
                vencedor = self.encontrar_vencedor(x)
                vizinhos = self.obter_vizinhos(vencedor)
                
                # Regra de atualização: w_j(t+1) = w_j(t) + eta * (x - w_j(t))
                for j in vizinhos:
                    self.W[j] += self.eta * (x - self.W[j])

    def mapear_classes_grid(self, X, rotulos):
        """
        Associa cada um dos 16 neurônios à classe que ele ativa no treinamento.
        Para neurônios de fronteira que não venceram diretamente, atribui a classe
        do protótipo mais próximo para permitir classificação completa.
        """
        contagem = {j: {'A': 0, 'B': 0, 'C': 0} for j in range(self.num_neuronios)}
        
        for x, r in zip(X, rotulos):
            vencedor = self.encontrar_vencedor(x)
            contagem[vencedor][r] += 1
            
        mapa_final = {}
        ativos_por_classe = {'A': [], 'B': [], 'C': []}
        
        # Centroides médios das classes para classificar neurônios inativos (se houver)
        centroides = {
            'A': np.mean(X[0:20], axis=0),
            'B': np.mean(X[20:60], axis=0),
            'C': np.mean(X[60:120], axis=0)
        }
        
        for j in range(self.num_neuronios):
            total_ativacoes = sum(contagem[j].values())
            num_neuronio = j + 1 # Índice de 1 a 16 conforme PDF
            if total_ativacoes > 0:
                classe_vencedora = max(contagem[j], key=contagem[j].get)
                mapa_final[num_neuronio] = classe_vencedora
                ativos_por_classe[classe_vencedora].append(num_neuronio)
            else:
                # Neurônio de transição/fronteira: associa pela menor distância ao centroide
                classe_prox = min(centroides.keys(), key=lambda c: np.linalg.norm(self.W[j] - centroides[c]))
                mapa_final[num_neuronio] = f"{classe_prox}*"
                
        return ativos_por_classe, mapa_final, contagem

# =====================================================================
# 3. EXECUÇÃO E RESPOSTAS DAS QUESTÕES 1 E 2
# =====================================================================

if __name__ == "__main__":
    som = RedeKohonen(linhas=4, colunas=4, dim_entrada=3, eta=0.001, raio=1)
    som.treinar(dados_treinamento, epocas=2000)

    ativos, mapa_grid, contagem = som.mapear_classes_grid(dados_treinamento, rotulos_treinamento)

    print("=" * 65)
    print("QUESTÃO 1: CONJUNTOS DE NEURÔNIOS POR CLASSE NO GRID (1 a 16)")
    print("=" * 65)
    for c in ['A', 'B', 'C']:
        print(f"Classe {c} (Ativados diretamente): Neurônios {ativos[c]}")

    print("\nVisualização do Grid Topológico 4x4 (Neurônio: Classe [Ativações]):")
    for l in range(4):
        linha_str = ""
        for col in range(4):
            idx = l * 4 + col
            n_id = idx + 1
            tot = sum(contagem[idx].values())
            linha_str += f"[{n_id:2d}: {mapa_grid[n_id]:2s} ({tot:2d})]  "
        print(linha_str)

    print("\n" + "=" * 65)
    print("QUESTÃO 2: CLASSIFICAÇÃO DAS 12 AMOSTRAS DE TESTE")
    print("=" * 65)
    print(f"{'Amostra':^8} | {'x1':^8} | {'x2':^8} | {'x3':^8} | {'Neurônio Vencedor':^18} | {'Classe':^6}")
    print("-" * 65)
    
    for i, x_teste in enumerate(amostras_teste, start=1):
        vencedor_idx = som.encontrar_vencedor(x_teste)
        neuronio_num = vencedor_idx + 1
        classe = mapa_grid[neuronio_num].replace("*", "")
        print(f"{i:^8d} | {x_teste[0]:^8.4f} | {x_teste[1]:^8.4f} | {x_teste[2]:^8.4f} | {neuronio_num:^18d} | {classe:^6s}")

    # Plotagem Gráfica do Mapa Topológico 4x4
    cores = {'A': '#4CAF50', 'B': '#2196F3', 'C': '#FF9800'}
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for l in range(4):
        for col in range(4):
            idx = l * 4 + col
            n_id = idx + 1
            classe_limpa = mapa_grid[n_id].replace("*", "")
            cor = cores[classe_limpa]
            tot = sum(contagem[idx].values())
            
            # Desenha conexões da grade
            if col < 3:
                ax.plot([col, col + 1], [3 - l, 3 - l], 'k-', lw=1.5, zorder=1)
            if l < 3:
                ax.plot([col, col], [3 - l, 3 - (l + 1)], 'k-', lw=1.5, zorder=1)
                
            # Desenha o neurônio
            circulo = plt.Circle((col, 3 - l), 0.35, color=cor, ec='black', lw=2, zorder=2, alpha=0.9 if tot > 0 else 0.4)
            ax.add_patch(circulo)
            ax.text(col, 3 - l, f"N{n_id}\nClasse {mapa_grid[n_id]}\n({tot} am.)", 
                    ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.6, 3.6)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Grid Topológico 4x4 da Rede de Kohonen\n(* indica neurônio de fronteira sem ativação direta)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()