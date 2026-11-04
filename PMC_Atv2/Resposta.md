4 - 
A superfície da função de custo no espaço de pesos de uma rede neural multicamadas é não convexa, caracterizada por múltiplos mínimos locais, pontos de sela e platôs com gradiente próximo de zero. Ao inicializar as matrizes de pesos com valores aleatórios distintos, o algoritmo se posiciona em diferentes pontos da bacia de atração. A trajetória percorrida pelo gradiente descendente varia a cada execução, convergindo para mínimos locais diferentes e exigindo caminhos mais longos ou curtos até atingir o critério de parada $\epsilon = 10^{-6}$.  
6 - A melhor configuração para o sistema de ressonância magnética deve equilibrar um baixo Erro Relativo Médio com uma baixa Variância no conjunto de validação (evitando oscilações bruscas nas predições).   O treinamento T2 alcança um erro relativo médio de 2,17% com a menor dispersão entre todas as configurações (variância de apenas 2,30%), demonstrando ser o modelo mais consistente, robusto e com a melhor capacidade de generalização para novos dados.  
  Dados
  =================================================================
Treinamento Erro Quadrático Médio  Número de Épocas
    1º (T1)          9.287632e-04               742
    2º (T2)          8.765904e-04               810
    3º (T3)          7.116889e-04              1006
    4º (T4)          1.499555e-03               213
    5º (T5)          1.019735e-03               713

==========================================================================================  
TABELA ITEM 5: VALIDAÇÃO COM O CONJUNTO DE TESTE
==========================================================================================
 | Amostra | $x_1$ | $x_2$ | $x_3$ | $d$ | $V_{rede}$ (T1) | $V_{rede}$ (T2) | $V_{rede}$ (T3) | $V_{rede}$ (T4) | $V_{rede}$ (T5) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.0611 | 0.2860 | 0.7464 | 0.4831 | 0.4974 | 0.4991 | 0.5007 | 0.4938 | 0.4978 |
| 2 | 0.5102 | 0.7464 | 0.0860 | 0.5965 | 0.6163 | 0.6087 | 0.6042 | 0.5965 | 0.6087 |
| 3 | 0.0004 | 0.6916 | 0.5006 | 0.5318 | 0.5094 | 0.5147 | 0.5160 | 0.5368 | 0.5176 |
| 4 | 0.9430 | 0.4476 | 0.2648 | 0.6843 | 0.7049 | 0.7077 | 0.7136 | 0.7114 | 0.7115 |
| 5 | 0.1399 | 0.1610 | 0.2477 | 0.2872 | 0.2745 | 0.2755 | 0.2772 | 0.2678 | 0.2717 |
| 6 | 0.6423 | 0.3229 | 0.8567 | 0.7663 | 0.7664 | 0.7629 | 0.7616 | 0.7532 | 0.7611 |
| 7 | 0.6492 | 0.0007 | 0.6422 | 0.5666 | 0.5639 | 0.5586 | 0.5463 | 0.5771 | 0.5588 |
| 8 | 0.1818 | 0.5078 | 0.9046 | 0.6601 | 0.6866 | 0.6857 | 0.6954 | 0.6836 | 0.6921 |
| 9 | 0.7382 | 0.2647 | 0.1916 | 0.5427 | 0.5364 | 0.5336 | 0.5424 | 0.5413 | 0.5401 |
| 10 | 0.3879 | 0.1307 | 0.8656 | 0.5836 | 0.6248 | 0.6151 | 0.6066 | 0.6097 | 0.6171 |
| 11 | 0.1903 | 0.6523 | 0.7820 | 0.6950 | 0.6897 | 0.6918 | 0.7015 | 0.6954 | 0.6961 |
| 12 | 0.8401 | 0.4490 | 0.2719 | 0.6790 | 0.6834 | 0.6831 | 0.6898 | 0.6814 | 0.6859 |
| 13 | 0.0029 | 0.3264 | 0.2476 | 0.2956 | 0.2808 | 0.2852 | 0.2840 | 0.2789 | 0.2789 |
| 14 | 0.7088 | 0.9342 | 0.2763 | 0.7742 | 0.7958 | 0.7933 | 0.7887 | 0.7770 | 0.7910 |
| 15 | 0.1283 | 0.1882 | 0.7253 | 0.4662 | 0.4809 | 0.4799 | 0.4783 | 0.4717 | 0.4777 |
| 16 | 0.8882 | 0.3077 | 0.8931 | 0.8093 | 0.8134 | 0.8152 | 0.8091 | 0.8132 | 0.8111 |
| 17 | 0.2225 | 0.9182 | 0.7820 | 0.7581 | 0.7538 | 0.7587 | 0.7670 | 0.7742 | 0.7655 |
| 18 | 0.1957 | 0.8423 | 0.3085 | 0.5826 | 0.5828 | 0.5840 | 0.5785 | 0.5977 | 0.5838 |
| 19 | 0.9991 | 0.5914 | 0.3933 | 0.7938 | 0.7992 | 0.8038 | 0.8077 | 0.7964 | 0.8026 |
| 20 | 0.2299 | 0.1524 | 0.7353 | 0.5012 | 0.5168 | 0.5122 | 0.5091 | 0.5042 | 0.5112 |

------------------------------------------------------------------------------------------
                Métrica    T1    T2    T3    T4    T5
Erro Relativo Médio (%) 2.40% 2.17% 2.27% 1.97% 2.31%
          Variância (%) 3.87% 2.30% 2.38% 3.88% 3.57%