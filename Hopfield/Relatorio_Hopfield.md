# Relatório e Respostas — Memória Associativa com Rede de Hopfield

---

## 1. Verificação de Conformidade do Código (`Hopfield.py`)


| **Dimensão e Neurônios:** Imagens de 45 bits em uma rede com 45 neurônios | Grade $9 \times 5 = 45$ pixels/neurônios (`num_neuronios=45` nas linhas 9–55 e 65) | Atendido |
| **Codificação Bipolar:** Pixel branco = `-1`, Pixel escuro = `+1` | Matrizes `padrao_1` a `padrao_4` codificadas exatamente com `-1` e `+1` conforme as figuras do PDF (linhas 9–55) | Atendido |
| **Topologia — Regra do Produto Externo:** Matriz de pesos $\mathbf{W}$ obtida via produto externo | Método `treinar_produto_externo` (linhas 70–83): calcula $\mathbf{W} = \frac{1}{N}\sum_{k=1}^{P} \mathbf{v}_k \mathbf{v}_k^T$ com diagonal principal zerada ($w_{ii} = 0$, sem auto-realimentação) | Atendido |
| **Topologia — Função de Ativação:** Tangente Hiperbólica com $\beta$ muito grande | Método `ativacao_tanh` (linhas 85–91): aplica $f(u) = \tanh(\beta \cdot u)$ com $\beta = 100.0$ e limiarização bipolar ($\pm 1$) | Atendido |
| **Ruído de Transmissão:** Cerca de 20% dos pixels corrompidos aleatoriamente (`-1` $\leftrightarrow$ `+1`) | Função `adicionar_ruido` (linhas 121–130): sorteia exatamente $20\% \times 45 = 9$ pixels sem reposição e inverte seus sinais | Atendido |
| **Atividades 1 e 2:** Simular 12 situações (3 para cada padrão) e exibir imagem transmitida, distorcida e recuperada | Laço principal (linhas 160–194): executa 3 simulações para cada um dos 4 dígitos, exibindo comparativo no terminal e gerando painel gráfico $12 \times 3$ em Matplotlib | Atendido |

---

## 2. Resultados das 12 Simulações (Atividades 1 e 2)

Com a semente fixa (`np.random.seed(42)`) e nível de ruído de **20% (9 bits invertidos em 45)**, a rede obteve **10/12 recuperações 100% exatas (83,3% de sucesso total)**:

* **Dígito 1 (3/3 sucessos):** Recuperação perfeita nas 3 simulações.
* **Dígito 2 (1/3 sucesso exato, 2/3 estado espúrio próximo):** Nas simulações 1 e 2, a rede limpou quase todo o ruído, mas convergiu para um atrator espúrio que difere do Dígito 2 em apenas 2 pixels (coluna direita das linhas 6 e 7), devido à altíssima sobreposição com o Dígito 3. Na simulação 3, recuperou 100% perfeitamente.
* **Dígito 3 (3/3 sucessos):** Recuperação perfeita nas 3 simulações.
* **Dígito 4 (3/3 sucessos):** Recuperação perfeita nas 3 simulações.

---

## 3. Resposta da Pergunta Solicitada (Atividade 3)

> **Pergunta 3:** *Explique o que acontece quando aumentamos excessivamente o nível de ruído.*

### Resposta:

Quando aumentamos excessivamente o nível de ruído nas imagens transmitidas, a **Rede de Hopfield perde a capacidade de recuperar fielmente o padrão original**, ocorrendo os seguintes fenômenos no espaço de estados da memória associativa:

1. **Saída da Bacia de Atração Original (Convergência para Outro Padrão Armazenado):**
   Cada padrão memorizado possui uma região ao seu redor no espaço de estados de 45 bits chamada **bacia de atração** (medida pela distância de Hamming, isto é, o número de bits diferentes). Com ruído moderado (~20%, ou 9 bits), o estado ruidoso permanece dentro da bacia de atração do padrão original e a dinâmica de minimização da função de energia de Lyapunov conduz a rede de volta ao padrão correto. Porém, quando o ruído é elevado (por exemplo, 35% a 45%), o vetor distorcido pode ficar mais próximo (em distância de Hamming) de **outro padrão memorizado** do que daquele que foi originalmente enviado — especialmente entre padrões correlacionados como os dígitos `2` e `3`. Nesse caso, a rede converge para o dígito errado.

2. **Convergência para Estados Espúrios (Mínimos Locais Indesejados):**
   Com alto nível de ruído, o estado inicial cai em regiões intermediárias do espaço de busca, onde a rede frequentemente estabiliza em **estados espúrios** (mínimos locais da função de energia que não correspondem a nenhum dos 4 padrões de treinamento). Esses estados espúrios costumam ser combinações lineares (misturas de pedaços de dois ou mais dígitos) ou estados parasitas gerados pela interferência cruzada (*crosstalk*) na matriz de pesos $\mathbf{W}$.

3. **Inversão de Fase (Convergência para o Padrão Reverso / Negativo Acima de 50% de Ruído):**
   A função de energia da Rede de Hopfield, $E = -\frac{1}{2}\mathbf{v}^T \mathbf{W} \mathbf{v}$, é **simétrica** em relação ao sinal do vetor de estado: $E(\mathbf{v}) = E(-\mathbf{v})$. Isso significa que, para cada padrão memorizado $\mathbf{v}_k$, o seu **inverso exato (negativo fotográfico)** $-\mathbf{v}_k$ também é um ponto fixo (atrator) com a mesma energia mínima.
   * Quando o nível de ruído ultrapassa **50%** (mais de 22 dos 45 pixels invertidos), a imagem corrompida passa a estar mais próxima do padrão invertido $-\mathbf{v}_k$ do que do padrão original $+\mathbf{v}_k$.
   * No limite de **80% a 100% de ruído**, a rede converge sistematicamente para a **imagem negativa** do dígito transmitido (onde tudo que era branco fica escuro e tudo que era escuro fica branco).

---

## 4. Conclusões

1. **Eficácia da Rede de Hopfield como Corretor de Ruído:**
   A rede implementada cumpriu com sucesso o papel de memória associativa autoassociativa para o canal de comunicação de 45 bits. Para um nível de ruído de 20% (9 pixels corrompidos), a dinâmica assíncrona com função de ativação $\tanh(\beta u)$ ($\beta \gg 1$) conseguiu restaurar a informação visual quase integralmente em todas as transmissões.

2. **Capacidade de Armazenamento vs. Correlação entre Padrões (*Crosstalk*):**
   Teoricamente, a capacidade máxima de armazenamento livre de erros de uma Rede de Hopfield treinada pela regra do produto externo (Hebbiana) para padrões aleatórios e ortogonais é de aproximadamente:
   $$P_{\max} \approx 0{,}138 \times N$$
   Para $N = 45$ neurônios, $P_{\max} \approx 6{,}2$ padrões. Embora tenhamos armazenado apenas $P = 4$ padrões (abaixo do limite teórico de 6), os padrões **Dígito 2** e **Dígito 3** não são ortogonais: eles possuem **37 pixels idênticos em 45 (82,2% de similaridade)**, diferindo apenas em 8 pixels nas linhas inferiores. Essa forte correlação gera interferência cruzada (*crosstalk*) na regra do produto externo, explicando por que o Dígito 2 apresentou um pequeno artefato de 2 pixels na borda inferior direita em duas simulações.

3. **Importância da Atualização Assíncrona e Diagonal Nula:**
   O zeramento da diagonal principal ($w_{ii} = 0$) evita que um neurônio reforce permanentemente seu próprio estado ruidoso inicial, enquanto a atualização assíncrona garante matematicamente que a função de energia da rede nunca aumente ($\Delta E \le 0$), assegurando convergência em poucas épocas sem oscilações de ciclo limite.
