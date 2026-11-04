# Centro Federal de Educação Tecnológica de Minas Gerais - CEFET-MG
## Campus VIII – Varginha | Bacharelado em Sistemas de Informação
**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Trabalho:** Redes de Função de Base Radial (RBF 1) - Classificação de Radiação  
**Data:** 04/11/2026  

---

### Descrição do Problema
A verificação da presença de radiação em determinados compostos nucleares é realizada a partir da concentração de duas variáveis de entrada ($x_1$ e $x_2$). A rede RBF implementada possui:
- **Camada de Entrada:** 2 variáveis ($x_1, x_2$).
- **Camada Intermediária:** 2 neurônios com funções de ativação gaussianas (representando os 2 clusters de presença de radiação).
- **Camada de Saída:** 1 neurônio com saída linear e pós-processamento via função sinal.
- **Padronização da Saída:**
  - Presença de Radiação: $+1$
  - Ausência de Radiação: $-1$

---

## Atividade 1: Treinamento da Camada Escondida (K-Means)

Conforme a especificação, o algoritmo $k$-means foi executado considerando **apenas os padrões do conjunto de treinamento com presença de radiação ($d = 1$)**, totalizando 19 amostras.

A variância ($\sigma_k^2$) de cada cluster foi calculada como a dispersão média quadrática das amostras pertencentes ao cluster em relação ao seu centroide:
$$\sigma_k^2 = \frac{1}{N_k} \sum_{x \in C_k} \|x - \mu_k\|^2$$

### Tabela 1: Coordenadas do Centro e Variância de cada Cluster

| Cluster | Centro $(x_1, x_2)$ | Variância ($\sigma^2$) |
| :---: | :---: | :---: |
| **1** | $(0.164833, \; 0.612117)$ | **0.029806** *(amostral: 0.035767)* |
| **2** | $(0.398969, \; 0.157131)$ | **0.038460** *(amostral: 0.041665)* |

*Nota: O Cluster 1 agrupou 6 amostras e o Cluster 2 agrupou 13 amostras.*

---

## Atividade 2: Treinamento da Camada de Saída (Regra Delta Generalizada)

* **Parâmetros:**
  - Taxa de aprendizado: $\eta = 0.01$
  - Precisão de parada: $\epsilon = 10^{-7}$ ($|EQM(t) - EQM(t-1)| \le 10^{-7}$)
  - Pesos iniciais: valores pseudoaleatórios entre $0$ e $1$.
* **Função de Base Radial Gaussiana:**
  $$\phi_k(x) = \exp\left(-\frac{\|x - \mu_k\|^2}{2\sigma_k^2}\right)$$
* **Convergência:** Atingida na **época 325**, com Erro Quadrático Médio final $EQM = 0.234239$.

### Tabela 2: Valores dos Pesos do Neurônio de Saída

| Peso | Valor ($x_0 = -1$, padrão CEFET/Adaline) | Valor ($x_0 = +1$, padrão tradicional) |
| :---: | :---: | :---: |
| **$W_{21,0}$** | **1.0027** | **-1.0027** |
| **$W_{21,1}$** | **2.3781** | **2.3781** |
| **$W_{21,2}$** | **2.6977** | **2.6977** |

---

## Atividade 3: Rotina de Pós-Processamento

Para converter a saída contínua da rede $y \in \mathbb{R}$ nos rótulos de classe inteiros $\{-1, 1\}$, implementou-se a função sinal (degrau bipolar):

$$y^{\text{pós}} = \begin{cases} +1, & \text{se } y \ge 0 \\ -1, & \text{se } y < 0 \end{cases}$$

Esta função é aplicada exclusivamente durante a etapa de avaliação/validação sobre os dados de teste.

---

## Atividade 4: Validação no Conjunto de Teste

Aplicando os pesos obtidos sobre o conjunto de teste fornecido (`teste_1.csv`):

### Tabela 3: Resultados de Validação

| Amostra | $x_1$ | $x_2$ | $d$ | $y$ (saída contínua) | $y^{\text{pós}}$ | Classificação |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.8705 | 0.9329 | -1 | -1.0025 | **-1** | Correto |
| **2** | 0.0388 | 0.2703 | 1 | -0.3231 | **-1** | Erro |
| **3** | 0.8236 | 0.4458 | -1 | -0.9140 | **-1** | Correto |
| **4** | 0.7075 | 0.1502 | 1 | -0.2201 | **-1** | Erro |
| **5** | 0.9587 | 0.8663 | -1 | -1.0026 | **-1** | Correto |
| **6** | 0.6115 | 0.9365 | -1 | -0.9878 | **-1** | Correto |
| **7** | 0.3534 | 0.3646 | 1 | 0.9665 | **1** | Correto |
| **8** | 0.3268 | 0.2766 | 1 | 1.3232 | **1** | Correto |
| **9** | 0.6129 | 0.4518 | -1 | -0.4682 | **-1** | Correto |
| **10** | 0.9948 | 0.4962 | -1 | -0.9966 | **-1** | Correto |

$$\text{Taxa de Acerto (\%)} = \frac{8}{10} \times 100\% = \mathbf{80.00\%}$$

---

## Atividade 5: Estratégias para Aumentar a Taxa de Acerto da RBF

A rede obteve 80% de acerto no teste, errando as amostras **2** ($x_1=0.0388, x_2=0.2703$) e **4** ($x_1=0.7075, x_2=0.1502$), ambas pertencentes à classe positiva ($d = 1$). As seguintes estratégias podem ser adotadas para mitigar essas falhas e elevar a taxa de acerto:

1. **Aumentar o número de neurônios da camada escondida ($K > 2$):**
   - Com apenas 2 centros gaussianos, a rede cobre apenas duas regiões hiperelípticas locais. Amostras periféricas da classe positiva acabam ficando distantes de ambos os centros e recebem uma ativação quase nula ($\phi_1 \approx 0, \phi_2 \approx 0$). Nessas regiões, o bias negativo predomina, classificando erroneamente como ausência de radiação. Usar 3 ou 4 centros cobriria essas zonas periféricas.

2. **Executar o K-Means sobre todo o conjunto de dados (ambas as classes):**
   - O agrupamento considerou exclusivamente os pontos com radiação ($d = 1$). Alocar centros gaussianos também para a classe sem radiação ($d = -1$) permite que a rede aprenda de forma balanceada a fronteira de separação entre ambas as distribuições.

3. **Treinamento Supervisionado Híbrido com Ajuste Fino (*Fine-Tuning*):**
   - Em vez de manter os centros ($\mu_k$) e as variâncias ($\sigma_k^2$) congelados após o $k$-means não-supervisionado, aplicar o algoritmo de gradiente descendente para ajustar dinamicamente tanto os pesos da saída quanto os centros e dispersões das gaussianas.

4. **Calibração da Largura das Gaussianas ($\sigma$):**
   - O raio de cobertura pode ser calibrado por um fator multiplicativo de espalhamento ($\sigma_{\text{novo}} = \alpha \cdot \sigma$, com $\alpha > 1$) ou pela heurística da distância entre centros ($\sigma = \frac{d_{\text{centros}}}{\sqrt{2K}}$), garantindo uma sobreposição suave que englobe pontos mais afastados.

---
*Gráficos de convergência e visualização do espaço de características disponíveis em:* `graficos_rbf1.png`.

