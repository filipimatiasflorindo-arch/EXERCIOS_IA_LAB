import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DEFINIÇÃO DOS PADRÕES (9 linhas x 5 colunas = 45 bits)
# Pixel branco = -1 | Pixel escuro = +1
# =====================================================================

padrao_1 = np.array([
    [-1, -1,  1,  1, -1],
    [-1,  1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1]
])

padrao_2 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1, -1, -1, -1],
    [ 1,  1, -1, -1, -1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1]
])

padrao_3 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1]
])

padrao_4 = np.array([
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1]
])

padroes = [padrao_1, padrao_2, padrao_3, padrao_4]
nomes_padroes = ["Dígito 1", "Dígito 2", "Dígito 3", "Dígito 4"]

# =====================================================================
# 2. IMPLEMENTAÇÃO DA REDE DE HOPFIELD
# =====================================================================

class RedeHopfield:
    def __init__(self, num_neuronios=45, beta=100.0):
        self.N = num_neuronios
        self.beta = beta
        self.W = np.zeros((self.N, self.N))

    def treinar_produto_externo(self, lista_padroes):
        """
        Obtém a matriz de pesos W pela regra do produto externo (Hebbiana),
        zerando a diagonal principal (sem auto-realimentação).
        """
        self.W = np.zeros((self.N, self.N))
        for p in lista_padroes:
            v = p.flatten().reshape(-1, 1)  # Vetor coluna (45, 1)
            self.W += np.dot(v, v.T)        # Produto externo: v * v^T
        
        # Zera a diagonal principal (w_ii = 0)
        np.fill_diagonal(self.W, 0)
        # Normalização opcional por N
        self.W /= self.N

    def ativacao_tanh(self, u):
        """
        Função de ativação Tangente Hiperbólica com Beta muito grande:
        f(u) = tanh(beta * u). Como beta é grande, tende a +1 ou -1.
        """
        val = np.tanh(self.beta * u)
        return np.where(val >= 0, 1, -1)

    def recuperar(self, padrao_ruidoso, max_epocas=100):
        """
        Recupera o padrão usando atualização assíncrona (neurônio por neurônio
        em ordem aleatória) até convergir para um ponto fixo (atrator).
        """
        estado = padrao_ruidoso.flatten().copy()
        
        for _ in range(max_epocas):
            estado_anterior = estado.copy()
            
            # Atualização assíncrona garante convergência na rede de Hopfield
            indices = np.random.permutation(self.N)
            for i in indices:
                u_i = np.dot(self.W[i, :], estado)
                # Se u_i == 0, mantém o estado atual do neurônio
                if u_i != 0:
                    estado[i] = self.ativacao_tanh(u_i)
            
            # Verifica convergência (estabilidade)
            if np.array_equal(estado, estado_anterior):
                break
                
        return estado.reshape(9, 5)

# =====================================================================
# 3. FUNÇÕES AUXILIARES (RUÍDO E IMPRESSÃO)
# =====================================================================

def adicionar_ruido(padrao, nivel_ruido=0.20):
    """
    Corrompe exatamente 20% dos pixels (9 de 45 pixels) aleatoriamente,
    invertendo seus sinais (-1 vira +1 e +1 vira -1).
    """
    v = padrao.flatten().copy()
    num_corrompidos = int(round(nivel_ruido * len(v))) # 20% de 45 = 9 pixels
    indices_sorteados = np.random.choice(len(v), size=num_corrompidos, replace=False)
    v[indices_sorteados] *= -1
    return v.reshape(9, 5)

def imprimir_comparacao_console(original, ruidoso, recuperado, titulo):
    """Exibe lado a lado no terminal o padrão original, distorcido e limpo."""
    mapa = {1: "██", -1: "░░"}
    print(f"\n--- {titulo} ---")
    print("Transmitida   Distorcida    Recuperada")
    for l in range(9):
        str_orig = "".join(mapa[val] for val in original[l])
        str_ruid = "".join(mapa[val] for val in ruidoso[l])
        str_rec  = "".join(mapa[val] for val in recuperado[l])
        print(f" {str_orig}    {str_ruid}    {str_rec}")

# =====================================================================
# 4. EXECUÇÃO DAS 12 SIMULAÇÕES (ATIVIDADES 1 E 2)
# =====================================================================

if __name__ == "__main__":
    # Fixa semente para reprodutibilidade dos testes
    np.random.seed(42)

    rede = RedeHopfield(num_neuronios=45, beta=100.0)
    rede.treinar_produto_externo(padroes)

    fig, axes = plt.subplots(12, 3, figsize=(7, 22))
    fig.suptitle("Simulações da Rede de Hopfield (12 Transmissões - 20% de Ruído)", fontsize=14, y=0.995)

    linha_plot = 0
    acertos = 0

    for idx_p, (padrao, nome) in enumerate(zip(padroes, nomes_padroes)):
        for tentativa in range(1, 4): # 3 situações para cada padrão
            img_distorcida = adicionar_ruido(padrao, nivel_ruido=0.20)
            img_recuperada = rede.recuperar(img_distorcida)

            sucesso = np.array_equal(padrao, img_recuperada)
            if sucesso:
                acertos += 1

            titulo_sim = f"{nome} - Simulação {tentativa} ({'Sucesso' if sucesso else 'Erro/Espúrio'})"
            imprimir_comparacao_console(padrao, img_distorcida, img_recuperada, titulo_sim)

            # Plotagem gráfica igual à folha do trabalho
            imagens = [padrao, img_distorcida, img_recuperada]
            titulos_col = ["Imagem Transmitida\n(livre de ruído)", 
                           "Imagem Distorcida\n(com 20% ruído)", 
                           "Imagem Limpa\n(recuperada)"]

            for col in range(3):
                ax = axes[linha_plot, col]
                ax.imshow(imagens[col], cmap="gray_r", vmin=-1, vmax=1)
                
                # Grade estilo pixel (9x5)
                ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
                ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8)
                ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

                if linha_plot == 0:
                    ax.set_title(titulos_col[col], fontsize=10, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f"{nome}\n(Sim. {tentativa})", fontsize=9, fontweight='bold')

            linha_plot += 1

    print(f"\nResumo: {acertos}/12 transmissões recuperadas perfeitamente.")
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.show()