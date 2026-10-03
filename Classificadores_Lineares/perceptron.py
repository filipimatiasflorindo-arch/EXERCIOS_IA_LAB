import numpy as np

# 1. Conjunto de Treinamento (30 padrões)
# Colunas: [x1, x2, x3, d]
dados_treino = np.array([
    [-0.6508,  0.1097,  4.0009, -1.0],
    [-1.4492,  0.8896,  4.4005, -1.0],
    [ 2.0850,  0.6876, 12.0710, -1.0],
    [ 1.1476,  0.2626,  7.7985,  1.0],
    [ 0.6418,  1.0234,  7.0427,  1.0],
    [ 0.6730,  0.2569,  8.3265, -1.0],
    [ 1.1155,  0.6043,  7.4446,  1.0],
    [ 0.3399,  0.0914,  7.0677, -1.0],
    [ 0.0121,  0.5256,  4.6316,  1.0],
    [-0.0429,  0.4660,  5.4323,  1.0],
    [ 0.4340,  0.6870,  8.2287, -1.0],
    [ 0.2735,  1.0287,  7.1934,  1.0],
    [ 0.4839,  0.4851,  7.4850, -1.0],
    [ 0.4089, -0.1267,  5.5019, -1.0],
    [ 1.4391,  0.1614,  8.5843, -1.0],
    [-0.9115, -0.1973,  2.1962, -1.0],
    [ 1.0475,  0.3654,  7.4858,  1.0],
    [ 0.2144,  0.7515,  7.1699,  1.0],
    [ 0.2013,  1.0014,  6.5489,  1.0],
    [ 0.6483,  0.2183,  5.8991,  1.0],
    [-0.1147,  0.2242,  7.2435, -1.0],
    [-0.7970,  0.8795,  3.8762,  1.0],
    [-1.0625,  0.6366,  2.4707,  1.0],
    [ 0.5307,  0.1285,  5.6883,  1.0],
    [-1.2200,  0.7777,  1.7252,  1.0],
    [ 0.3957,  0.1076,  5.6623, -1.0],
    [-0.1013,  0.5989,  7.1812, -1.0],
    [ 2.4482,  0.9455, 11.2095,  1.0],
    [ 2.0149,  0.6192, 10.9263, -1.0],
    [ 0.2012,  0.2611,  5.4631,  1.0]
])

# Amostras de Teste para o Item 3
dados_teste = np.array([
    [-0.3565,  0.0620,  5.9891],
    [-0.7842,  1.1267,  5.5912],
    [ 0.3012,  0.5611,  5.8234],
    [ 0.7757,  1.0648,  8.0677],
    [ 0.1570,  0.8028,  6.3040],
    [-0.7014,  1.0316,  3.6005],
    [ 0.3748,  0.1536,  6.1537],
    [-0.6920,  0.9404,  4.4058],
    [-1.3970,  0.7141,  4.9263],
    [-1.8842, -0.2805,  1.2548]
])

eta = 0.01

# Matriz X com x0 = -1 na primeira coluna
X_treino = np.column_stack([-np.ones(len(dados_treino)), dados_treino[:, 0:3]])
d_treino = dados_treino[:, 3]

X_teste = np.column_stack([-np.ones(len(dados_teste)), dados_teste])

def funcao_ativacao(u):
    return 1 if u >= 0 else -1

def treinar_perceptron(X, d, eta):
    # Inicialização aleatória dos pesos no intervalo [0, 1]
    w = np.random.uniform(0.0, 1.0, size=4)
    w_inicial = w.copy()
    
    epocas = 0
    max_epocas = 10000
    
    while epocas < max_epocas:
        epocas += 1
        houve_erro = False
        
        for i in range(len(X)):
            u = np.dot(w, X[i])
            y = funcao_ativacao(u)
            erro = d[i] - y
            
            if erro != 0:
                w += eta * erro * X[i]
                houve_erro = True
                
        if not houve_erro:
            break
            
    return w_inicial, w, epocas

# Executar 5 treinamentos (Item 1 e 2)
pesos_finais = []
print("--- RESULTADOS DOS TREINAMENTOS (ITEM 2) ---")
for t in range(1, 6):
    w_ini, w_fim, epocas = treinar_perceptron(X_treino, d_treino, eta)
    pesos_finais.append(w_fim)
    print(f"Treinamento {t} (T{t}):")
    print(f"  W_inicial: {np.round(w_ini, 4)}")
    print(f"  W_final:   {np.round(w_fim, 4)}")
    print(f"  Épocas:    {epocas}\n")

# Classificar amostras de teste (Item 3)
print("--- CLASSIFICAÇÃO DAS AMOSTRAS (ITEM 3) ---")
predicoes = np.zeros((len(dados_teste), 5))
for j, w in enumerate(pesos_finais):
    for i in range(len(dados_teste)):
        u = np.dot(w, X_teste[i])
        predicoes[i, j] = funcao_ativacao(u)

for i in range(len(dados_teste)):
    saidas = [f"{int(predicoes[i, t]):+d}" for t in range(5)]
    print(f"Amostra {i+1:02d}: " + " | ".join(saidas))