import numpy as np
import matplotlib.pyplot as plt

# 1. Dados de Treinamento (35 amostras: [x1, x2, x3, x4, d])
dados_treino = np.array([
    [ 0.4329, -1.3719,  0.7022, -0.8535,  1.0],
    [ 0.3024,  0.2286,  0.8630,  2.7909, -1.0],
    [ 0.1349, -0.6445,  1.0530,  0.5687, -1.0],
    [ 0.3374, -1.7163,  0.3670, -0.6283, -1.0],
    [ 1.1434, -0.0485,  0.6637,  1.2606,  1.0],
    [ 1.3749, -0.5071,  0.4464,  1.3009,  1.0],
    [ 0.7221, -0.7587,  0.7681, -0.5592,  1.0],
    [ 0.4403, -0.8072,  0.5154, -0.3129,  1.0],
    [-0.5231,  0.3548,  0.2538,  1.5776, -1.0],
    [ 0.3255, -2.0000,  0.7112, -1.1209,  1.0],
    [ 0.5824,  1.3915, -0.2291,  4.1735, -1.0],
    [ 0.1340,  0.6081,  0.4450,  3.2230, -1.0],
    [ 0.1480, -0.2988,  0.4778,  0.8649,  1.0],
    [ 0.7359,  0.1869, -0.0872,  2.3584,  1.0],
    [ 0.7115, -1.1469,  0.3394,  0.9573, -1.0],
    [ 0.8251, -1.2840,  0.8452,  1.2382, -1.0],
    [ 0.1569,  0.3712,  0.8825,  1.7633,  1.0],
    [ 0.0033,  0.6835,  0.5389,  2.8249, -1.0],
    [ 0.4243,  0.8313,  0.2634,  3.5855, -1.0],
    [ 1.0490,  0.1326,  0.9138,  1.9792,  1.0],
    [ 1.4276,  0.5331, -0.0145,  3.7286,  1.0],
    [ 0.5971,  1.4865,  0.2904,  4.6069, -1.0],
    [ 0.8475,  2.1479,  0.3179,  5.8235, -1.0],
    [ 1.3967, -0.4171,  0.6443,  1.3927,  1.0],
    [ 0.0044,  1.5378,  0.6099,  4.7755, -1.0],
    [ 0.2201, -0.5668,  0.0515,  0.7829,  1.0],
    [ 0.6300, -1.2480,  0.8591,  0.8093, -1.0],
    [-0.2479,  0.8960,  0.0547,  1.7381,  1.0],
    [-0.3088, -0.0929,  0.8659,  1.5483, -1.0],
    [-0.5180,  1.4974,  0.5453,  2.3993,  1.0],
    [ 0.6833,  0.8266,  0.0829,  2.8864,  1.0],
    [ 0.4353, -1.4066,  0.4207, -0.4879,  1.0],
    [-0.1069, -3.2329,  0.1856, -2.4572, -1.0],
    [ 0.4662,  0.6261,  0.7304,  3.4370, -1.0],
    [ 0.8298, -1.4089,  0.3119,  1.3235, -1.0]
])

# 2. Dados de Teste (15 amostras: [x1, x2, x3, x4])
dados_teste = np.array([
    [ 0.9694,  0.6909,  0.4334, 3.4965],
    [ 0.5427,  1.3832,  0.6390, 4.0352],
    [ 0.6081, -0.9196,  0.1016, 0.5925],
    [-0.1618,  0.4694,  3.0117, 0.2030],
    [ 0.1870, -0.2578,  0.6124, 1.7749],
    [ 0.4891, -0.5276,  0.6439, 0.4378],
    [ 0.3777,  2.0149,  3.3932, 0.7423],
    [ 1.1498, -0.4067,  0.2469, 1.5866],
    [ 0.9325,  1.0950,  1.0359, 3.3591],
    [ 0.5060,  1.3317,  3.7174, 0.9222],
    [ 0.0497, -2.0656, -0.6585, 0.6124],
    [ 0.4004,  3.5369,  5.3532, 0.9766],
    [-0.1874,  1.3343,  3.2189, 0.5374],
    [ 0.5060,  1.3317,  3.7174, 0.9222],
    [ 1.6375, -0.7911,  0.5515, 0.7537]
])

# Matriz estendida: coluna 0 é o bias fixado em x0 = -1
X_treino = np.column_stack([-np.ones(len(dados_treino)), dados_treino[:, 0:4]])
d_treino = dados_treino[:, 4]
X_teste  = np.column_stack([-np.ones(len(dados_teste)), dados_teste])

eta = 0.0025
precisao = 1e-6
N = len(X_treino)

def calcular_eqm(w, X, d):
    u = np.dot(X, w)
    return np.mean((d - u) ** 2)

def treinar_adaline(X, d, eta, precisao):
    w = np.random.uniform(0.0, 1.0, size=5)
    w_inicial = w.copy()
    
    historico_eqm = []
    eqm_anterior = calcular_eqm(w, X, d)
    historico_eqm.append(eqm_anterior)
    
    epocas = 0
    while True:
        epocas += 1
        # Atualização estocástica da Regra Delta (padrão a padrão)
        for i in range(len(X)):
            u_i = np.dot(w, X[i])
            erro = d[i] - u_i
            w += eta * erro * X[i]
            
        eqm_atual = calcular_eqm(w, X, d)
        historico_eqm.append(eqm_atual)
        
        if abs(eqm_atual - eqm_anterior) <= precisao:
            break
        eqm_anterior = eqm_atual
        
    return w_inicial, w, epocas, historico_eqm

# Execução dos 5 Treinamentos
pesos_finais = []
historicos_eqm = []

print("=== RESULTADOS DOS 5 TREINAMENTOS (ITEM 2) ===")
for t in range(1, 6):
    w_ini, w_fim, ep, hist = treinar_adaline(X_treino, d_treino, eta, precisao)
    pesos_finais.append(w_fim)
    historicos_eqm.append(hist)
    print(f"\nTreinamento {t} (T{t}):")
    print(f"  W_inicial: {np.round(w_ini, 4)}")
    print(f"  W_final:   {np.round(w_fim, 4)}")
    print(f"  Épocas:    {ep}")

# Item 3: Gráficos de EQM para T1 e T2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(historicos_eqm[0], color='tab:blue', lw=2)
ax1.set_title("EQM vs Épocas - Treinamento 1 (T1)")
ax1.set_xlabel("Época")
ax1.set_ylabel("Erro Quadrático Médio (EQM)")
ax1.grid(True, ls="--", alpha=0.6)

ax2.plot(historicos_eqm[1], color='tab:orange', lw=2)
ax2.set_title("EQM vs Épocas - Treinamento 2 (T2)")
ax2.set_xlabel("Época")
ax2.set_ylabel("Erro Quadrático Médio (EQM)")
ax2.grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.show()

# Item 4: Classificação das Amostras de Teste
print("\n=== CLASSIFICAÇÃO DAS AMOSTRAS DE TESTE (ITEM 4) ===")
print("Amostra | T1 | T2 | T3 | T4 | T5 | Válvula Indicada")
for i in range(len(dados_teste)):
    saidas = []
    for j in range(5):
        u = np.dot(pesos_finais[j], X_teste[i])
        y = 1 if u >= 0 else -1
        saidas.append(y)
    valvula = "Válvula B (+1)" if saidas[0] == 1 else "Válvula A (-1)"
    valores_fmt = " | ".join([f"{s:+d}" for s in saidas])
    print(f"{i+1:02d}      | {valores_fmt} | {valvula}")