# Trabalho: Rede de Kohonen (SOM)
**Instituição:** Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG) – Campus VIII (Varginha)  
**Curso:** Bacharelado em Sistemas de Informação  
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  

---

## Parâmetros do Treinamento
- **Arquitetura do Grid Topológico:** Bidimensional $4 \times 4$ ($N_1 = 16$ neurônios, numerados de $1$ a $16$)
- **Dimensão de Entrada:** $3$ grandezas $\{x_1, x_2, x_3\}$
- **Taxa de Aprendizado ($\eta$):** $0.001$
- **Raio de Vizinhança:** $r = 1$ (vizinhança de Von Neumann / conexões horizontais e verticais diretas)
- **Conjunto de Treinamento:** $120$ amostras (Apêndice), distribuídas em:
  - **Classe A:** Amostras $1$ a $20$ ($20$ amostras)
  - **Classe B:** Amostras $21$ a $60$ ($40$ amostras)
  - **Classe C:** Amostras $61$ a $120$ ($60$ amostras)

---

## Questão 1
> **Indique quem são os conjuntos de neurônios representados no grid que fornecem respostas relativas às classes A, B e C.**

Após o treinamento da Rede de Kohonen (com semente fixa `seed = 42` e $2000$ épocas em `Kohonen.py`), os neurônios do grid $4 \times 4$ agruparam-se topologicamente da seguinte forma:

### Conjuntos de Neurônios por Classe
| Classe | Neurônios Ativados Diretamente | Neurônios de Fronteira / Transição (Mais Próximos) | Conjunto Completo no Grid |
| :---: | :---: | :---: | :---: |
| **Classe A** | $\{11, 15, 16\}$ | — | $\{11, 15, 16\}$ |
| **Classe B** | $\{3, 4, 7, 8\}$ | $\{12\}$ | $\{3, 4, 7, 8, 12\}$ |
| **Classe C** | $\{1, 2, 5, 6, 9, 13\}$ | $\{10, 14\}$ | $\{1, 2, 5, 6, 9, 10, 13, 14\}$ |

### Disposição no Grid Topológico $4 \times 4$
Abaixo está a representação espacial do grid $4 \times 4$ indicando o número do neurônio, a classe associada e o total de amostras de treinamento para as quais cada neurônio foi o vencedor (BMU):

| Coluna 1 | Coluna 2 | Coluna 3 | Coluna 4 |
| :---: | :---: | :---: | :---: |
| **N1:** Classe **C** *(11 am.)* | **N2:** Classe **C** *(11 am.)* | **N3:** Classe **B** *(2 am.)* | **N4:** Classe **B** *(18 am.)* |
| **N5:** Classe **C** *(11 am.)* | **N6:** Classe **C** *(6 am.)* | **N7:** Classe **B** *(1 am.)* | **N8:** Classe **B** *(19 am.)* |
| **N9:** Classe **C** *(7 am.)* | **N10:** Classe **C\*** *(0 am.)* | **N11:** Classe **A** *(5 am.)* | **N12:** Classe **B\*** *(0 am.)* |
| **N13:** Classe **C** *(14 am.)* | **N14:** Classe **C\*** *(0 am.)* | **N15:** Classe **A** *(6 am.)* | **N16:** Classe **A** *(9 am.)* |

*(\* Neurônios situados na fronteira entre agrupamentos que atuaram como unidades de transição sem vencer diretamente no treinamento, classificados pelo centroide mais próximo).*

---

## Questão 2
> **Para as amostras da tabela abaixo indique a que classes as mesmas pertencem.**

Submetendo as $12$ amostras de teste à rede treinada e determinando o neurônio vencedor (menor distância Euclidiana $\|\mathbf{x} - \mathbf{w}_j\|$), obtém-se a seguinte classificação:

| Amostra | $x_1$ | $x_2$ | $x_3$ | Neurônio Vencedor (BMU) | Classe |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.2471$ | $0.1778$ | $0.2905$ | Neurônio $11$ | **A** |
| **2** | $0.8240$ | $0.2223$ | $0.7041$ | Neurônio $8$ | **B** |
| **3** | $0.4960$ | $0.7231$ | $0.5866$ | Neurônio $6$ | **C** |
| **4** | $0.2923$ | $0.2041$ | $0.2234$ | Neurônio $11$ | **A** |
| **5** | $0.8118$ | $0.2668$ | $0.7484$ | Neurônio $8$ | **B** |
| **6** | $0.4837$ | $0.8200$ | $0.4792$ | Neurônio $5$ | **C** |
| **7** | $0.3248$ | $0.2629$ | $0.2375$ | Neurônio $11$ | **A** |
| **8** | $0.7209$ | $0.2116$ | $0.7821$ | Neurônio $4$ | **B** |
| **9** | $0.5259$ | $0.6522$ | $0.5957$ | Neurônio $2$ | **C** |
| **10** | $0.2075$ | $0.1669$ | $0.1745$ | Neurônio $15$ | **A** |
| **11** | $0.7830$ | $0.3171$ | $0.7888$ | Neurônio $4$ | **B** |
| **12** | $0.5393$ | $0.7510$ | $0.5682$ | Neurônio $2$ | **C** |

---

## Questão 3
> **Demonstrar que a regra de alteração de pesos "Norma Euclidiana" para um padrão $\mathbf{x}$ é obtida a partir da minimização da função erro quadrático:**
> $$E = \frac{1}{2} \sum_{i} (w_{ji} - x_i)^2$$
> **onde $j$ é o índice do neurônio vencedor.**

### Demonstração

Seja $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$ o vetor de entrada apresentado à rede e $\mathbf{w}_j = [w_{j1}, w_{j2}, \dots, w_{jn}]^T$ o vetor de pesos sinápticos associado ao neurônio vencedor $j$.

**1. Definição da função objetivo (erro quadrático):**  
A função de custo $E$ mede a metade do quadrado da distância Euclidiana entre o padrão de entrada $\mathbf{x}$ e o vetor de pesos $\mathbf{w}_j$:

$$E(\mathbf{w}_j) = \frac{1}{2} \|\mathbf{w}_j - \mathbf{x}\|^2 = \frac{1}{2} \sum_{i=1}^{n} (w_{ji} - x_i)^2$$

**2. Aplicação do método do Gradiente Descendente:**  
Para minimizar a função de erro $E(\mathbf{w}_j)$ iterativamente, ajustamos cada componente de peso $w_{ji}$ na direção oposta ao gradiente de $E$ em relação a $w_{ji}$, proporcionalmente à taxa de aprendizado $\eta > 0$:

$$\Delta w_{ji} = -\eta \frac{\partial E}{\partial w_{ji}}$$

**3. Cálculo da derivada parcial:**  
Derivando $E$ em relação a uma componente específica $w_{ji}$ (aplicando a regra da cadeia, visto que os demais termos do somatório para $k \neq i$ são constantes em relação a $w_{ji}$):

$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left[ \frac{1}{2} \sum_{k=1}^{n} (w_{jk} - x_k)^2 \right]$$

$$\frac{\partial E}{\partial w_{ji}} = \frac{1}{2} \cdot 2 (w_{ji} - x_i) \cdot \frac{\partial}{\partial w_{ji}}(w_{ji} - x_i)$$

$$\frac{\partial E}{\partial w_{ji}} = (w_{ji} - x_i) \cdot 1 = w_{ji} - x_i$$

**4. Substituição na variação dos pesos ($\Delta w_{ji}$):**  
Substituindo a derivada parcial encontrada na equação do gradiente descendente:

$$\Delta w_{ji} = -\eta (w_{ji} - x_i) = \eta (x_i - w_{ji})$$

**5. Equação de atualização de pesos de Kohonen:**  
Portanto, o novo peso $w_{ji}(t+1)$ na iteração seguinte é dado por:

$$w_{ji}(t+1) = w_{ji}(t) + \Delta w_{ji}$$

$$w_{ji}(t+1) = w_{ji}(t) + \eta \left( x_i - w_{ji}(t) \right)$$

Ou, em notação vetorial para o neurônio vencedor $j$:

$$\mathbf{w}_j(t+1) = \mathbf{w}_j(t) + \eta \left( \mathbf{x} - \mathbf{w}_j(t) \right) \quad \blacksquare$$
