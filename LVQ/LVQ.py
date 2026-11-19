from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

# 1. Dados de Treinamento (7h, 8h, 9h, 10h, 11h, 12h) carregados de treinamento.csv
train_data = np.loadtxt(BASE_DIR / "treinamento.csv", delimiter=",", skiprows=1)
X_train = train_data[:, 1:7]
y_train = train_data[:, 7].astype(int)

# 2. Dados de Teste (Dias 1 a 8) carregados de teste.csv
test_data = np.loadtxt(BASE_DIR / "teste.csv", delimiter=",", skiprows=1)
X_test = test_data[:, 1:7]

class LVQ1:
    def __init__(self, gamma=0.05, epochs=100, decay=True):
        self.gamma = gamma
        self.epochs = epochs
        self.decay = decay
        self.prototypes = None
        self.prototype_labels = None

    def fit(self, X, y):
        classes = np.unique(y)
        self.prototype_labels = classes.copy()
        
        # Inicializa 1 protótipo por classe usando a primeira amostra de cada classe
        self.prototypes = np.array([X[y == c][0].copy() for c in classes])
        
        for epoch in range(self.epochs):
            # Decaimento linear da taxa de aprendizagem (opcional, melhora convergência)
            lr = self.gamma * (1 - epoch / self.epochs) if self.decay else self.gamma
            
            for x_i, target in zip(X, y):
                # Calcula a distância euclidiana de x_i para todos os protótipos
                distances = np.linalg.norm(self.prototypes - x_i, axis=1)
                
                # Encontra o índice do protótipo vencedor (BMU - Best Matching Unit)
                winner_idx = np.argmin(distances)
                
                # Atualiza os pesos do protótipo vencedor conforme regra LVQ-1
                if self.prototype_labels[winner_idx] == target:
                    self.prototypes[winner_idx] += lr * (x_i - self.prototypes[winner_idx])
                else:
                    self.prototypes[winner_idx] -= lr * (x_i - self.prototypes[winner_idx])

    def predict(self, X):
        predictions = []
        for x_i in X:
            distances = np.linalg.norm(self.prototypes - x_i, axis=1)
            winner_idx = np.argmin(distances)
            predictions.append(self.prototype_labels[winner_idx])
        return np.array(predictions)

# 3. Instanciar, Treinar e Avaliar a Rede LVQ-1
lvq = LVQ1(gamma=0.05, epochs=100, decay=True)
lvq.fit(X_train, y_train)

# Verificando acurácia no treinamento
train_preds = lvq.predict(X_train)
acc_train = np.mean(train_preds == y_train) * 100
print(f"Acurácia no Treinamento: {acc_train:.2f}%\n")

# Exibindo os vetores protótipos finais aprendidos para cada classe
print("Protótipos (Pesos Finais) por Classe:")
for label, proto in zip(lvq.prototype_labels, lvq.prototypes):
    print(f"Classe {label}: {np.round(proto, 4)}")

# 4. Classificação dos Dados de Teste
test_preds = lvq.predict(X_test)

print("\n--- Resultados da Classificação (Dados de Teste) ---")
for dia, (x_i, classe) in enumerate(zip(X_test, test_preds), start=1):
    dists = np.linalg.norm(lvq.prototypes - x_i, axis=1)
    min_dist = np.min(dists)
    print(f"Dia {dia}: Classe {classe} (distância euclidiana = {min_dist:.4f})")


def gerar_graficos(lvq_model, X_tr, y_tr, X_te, y_te_pred, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    horas = ["7h", "8h", "9h", "10h", "11h", "12h"]
    cores = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd"}
    y_max = max(X_tr.max(), X_te.max()) + 0.5

    # Gráfico 1: Padrões de Treinamento e Protótipo Aprendido por Classe
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, lvq_model.prototype_labels):
        amostras = X_tr[y_tr == c]
        idxs = np.where(y_tr == c)[0] + 1
        for idx, amostra in zip(idxs, amostras):
            ax.plot(horas, amostra, marker="o", alpha=0.45, linewidth=1.5,
                    color=cores[c], label=f"Amostra {idx}")
        proto = lvq_model.prototypes[lvq_model.prototype_labels == c][0]
        ax.plot(horas, proto, marker="s", linewidth=2.8, linestyle="--",
                color="black", label=f"Protótipo Classe {c}")
        ax.set_title(f"Classe {c} — Treinamento vs. Protótipo LVQ-1", fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper left")
    fig.supxlabel("Horário")
    fig.supylabel("Potência Elétrica")
    fig.suptitle("Padrões de Demanda de Treinamento e Protótipos LVQ-1", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "padroes_treinamento.png", dpi=150)
    plt.close(fig)

    # Gráfico 2: Comparação dos 4 Protótipos (Perfis de Demanda)
    fig, ax = plt.subplots(figsize=(9, 5))
    for c, proto in zip(lvq_model.prototype_labels, lvq_model.prototypes):
        ax.plot(horas, proto, marker="o", linewidth=2.5, color=cores[c], label=f"Classe {c}")
    ax.set_title("Comparação dos Protótipos Finais Aprendidos (LVQ-1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Horário")
    ax.set_ylabel("Potência Elétrica")
    ax.set_ylim(0, y_max)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "prototipos_classes.png", dpi=150)
    plt.close(fig)

    # Gráfico 3: Dias de Teste Classificados vs. Protótipos
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, lvq_model.prototype_labels):
        proto = lvq_model.prototypes[lvq_model.prototype_labels == c][0]
        ax.plot(horas, proto, marker="s", linewidth=2.6, linestyle="--",
                color="black", label=f"Protótipo Classe {c}")
        dias_idx = np.where(y_te_pred == c)[0]
        for i, idx in enumerate(dias_idx):
            estilo = "-" if i % 2 == 0 else "-."
            ax.plot(horas, X_te[idx], marker="o", linewidth=2.0, linestyle=estilo,
                    color=cores[c], alpha=0.85, label=f"Dia {idx + 1}")
        ax.set_title(f"Classe {c} — Dias de Teste Classificados", fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper left")
    fig.supxlabel("Horário")
    fig.supylabel("Potência Elétrica")
    fig.suptitle("Classificação dos Dados de Teste (Dias 1 a 8) via LVQ-1", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "classificacao_teste.png", dpi=150)
    plt.close(fig)


gerar_graficos(lvq, X_train, y_train, X_test, test_preds, BASE_DIR / "graficos")
print(f"\nGráficos salvos em: {BASE_DIR / 'graficos'}")