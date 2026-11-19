"""
Implementação da Rede Neural ART-1 (Adaptive Resonance Theory 1)
Atividade: Agrupamento de situações de um processo industrial (ART1.pdf)
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


class ART1:
    """Rede neural ART-1 para agrupamento de padrões binários."""

    def __init__(
        self,
        n_features: int,
        max_clusters: int,
        vigilance: float,
        L: float = 2.0,
        max_epochs: int = 100,
    ) -> None:
        if not (0.0 < vigilance <= 1.0):
            raise ValueError("O parâmetro de vigilância (rho) deve estar em (0, 1].")
        if L <= 1.0:
            raise ValueError("O parâmetro L deve ser maior que 1.")

        self.n_features = n_features
        self.max_clusters = max_clusters
        self.vigilance = vigilance
        self.L = L
        self.max_epochs = max_epochs

        # Pesos bottom-up (F1 -> F2): b_ij(0) = 1 / (1 + n)
        self.B = np.full(
            (n_features, max_clusters),
            1.0 / (1.0 + n_features),
            dtype=np.float64,
        )
        # Pesos top-down (F2 -> F1): t_ji(0) = 1
        self.T = np.ones((max_clusters, n_features), dtype=np.float64)
        self.epochs_trained = 0

    def _compute_activations(self, x: np.ndarray) -> np.ndarray:
        """Calcula a entrada líquida y_j = sum_i (b_ij * x_i) de forma determinística."""
        return np.array(
            [np.dot(x, self.B[:, j]) for j in range(self.max_clusters)],
            dtype=np.float64,
        )

    def _find_winning_cluster(self, x: np.ndarray, update_weights: bool) -> int:
        """Executa competição em F2 e teste de vigilância para um padrão de entrada x."""
        y = self._compute_activations(x)
        disabled = np.zeros(self.max_clusters, dtype=bool)
        norm_x = np.sum(x)

        while True:
            y_masked = np.where(disabled, -np.inf, y)
            winner = int(np.argmax(y_masked))

            if np.isinf(y_masked[winner]):
                raise RuntimeError(
                    "Número máximo de classes (max_clusters) insuficiente para classificar a entrada."
                )

            # Vetor de comparação s* = x AND t_winner
            s_star = x * self.T[winner]
            similarity = 1.0 if norm_x == 0 else float(np.sum(s_star) / norm_x)

            # Teste de vigilância
            if similarity >= self.vigilance:
                if update_weights:
                    self.T[winner] = s_star
                    self.B[:, winner] = (self.L * self.T[winner]) / (
                        self.L - 1.0 + np.sum(self.T[winner])
                    )
                return winner

            # Reset: inibe o neurônio vencedor e tenta o próximo candidato
            disabled[winner] = True

    def fit(self, X: np.ndarray) -> "ART1":
        """Treina a rede ART-1 até a convergência dos pesos ou atingir max_epochs."""
        X = np.asarray(X, dtype=np.float64)

        for epoch in range(1, self.max_epochs + 1):
            T_prev = self.T.copy()
            for x in X:
                self._find_winning_cluster(x, update_weights=True)

            self.epochs_trained = epoch
            if np.array_equal(self.T, T_prev):
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classifica os padrões de entrada sem alterar os pesos aprendidos."""
        X = np.asarray(X, dtype=np.float64)
        labels = [self._find_winning_cluster(x, update_weights=False) for x in X]
        return np.array(labels, dtype=int)


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega identificadores de situação, matriz binária e nomes das variáveis."""
    df = pd.read_csv(csv_path)
    situations = df.iloc[:, 0].to_numpy(dtype=int)
    feature_names = list(df.columns[1:])
    features = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    return situations, features, feature_names


def plot_active_classes(results: list[dict], out_dir: Path) -> Path:
    """Gráfico de barras com o número de classes ativas para cada grau de vigilância."""
    rhos = [str(r["rho"]) for r in results]
    n_classes = [r["n_active"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    colors = ["#2563eb", "#0891b2", "#0d9488", "#059669"]
    bars = ax.bar(rhos, n_classes, color=colors, width=0.52, edgecolor="#1e293b", linewidth=0.8)

    ax.axhline(10, color="#64748b", linestyle="--", linewidth=1.1, label="Total de situações (10)")

    for bar, val in zip(bars, n_classes):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.25,
            f"{val} classes",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    ax.set_title("Quantidade de Classes Ativas por Grau de Vigilância ($\\rho$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Grau de vigilância ($\\rho$)", fontsize=11)
    ax.set_ylabel("Número de classes ativas", fontsize=11)
    ax.set_ylim(0, 11.2)
    ax.set_yticks(range(0, 11, 2))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    out_path = out_dir / "classes_ativas_por_rho.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_cluster_assignments(results: list[dict], situations: np.ndarray, out_dir: Path) -> Path:
    """Grade 2x2 mostrando a qual classe cada situação pertence em cada valor de rho."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=150)
    palette = plt.get_cmap("tab10")

    for ax, res in zip(axes.flat, results):
        rho = res["rho"]
        active_clusters = res["active_clusters"]

        # Mapeia situação -> número sequencial da classe ativa (1..K)
        sit_to_class = {}
        for display_idx, (_, members) in enumerate(active_clusters.items(), start=1):
            for s in members:
                sit_to_class[s] = display_idx

        y_pos = np.arange(len(situations))
        classes = [sit_to_class[int(s)] for s in situations]
        bar_colors = [palette((c - 1) % 10) for c in classes]

        bars = ax.barh(y_pos, classes, color=bar_colors, edgecolor="#1e293b", linewidth=0.7, height=0.68)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Situação {int(s)}" for s in situations], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(classes) + 1.4)
        ax.set_xticks(range(1, max(classes) + 1))
        ax.set_xlabel("Classe ativa atribuída", fontsize=9.5)
        ax.set_title(
            f"$\\rho = {rho}$  ({res['n_active']} classes ativas)",
            fontsize=11.5,
            fontweight="bold",
            pad=8,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.55)
        ax.set_axisbelow(True)

        for bar, c in zip(bars, classes):
            ax.text(
                c + 0.12,
                bar.get_y() + bar.get_height() / 2.0,
                f"C{c}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#0f172a",
            )

    fig.suptitle("Agrupamento das Situações por Grau de Vigilância ($\\rho$)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "agrupamentos_por_rho.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_prototypes(results: list[dict], feature_names: list[str], out_dir: Path) -> Path:
    """Grade 2x2 com o mapa de calor dos vetores protótipo (T) aprendidos em cada rho."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    cmap = ListedColormap(["#f1f5f9", "#1d4ed8"])

    for ax, res in zip(axes.flat, results):
        rho = res["rho"]
        active_clusters = res["active_clusters"]
        T = res["T"]

        proto_rows = []
        y_labels = []
        for display_idx, (neuron_idx, members) in enumerate(active_clusters.items(), start=1):
            proto_rows.append(T[neuron_idx])
            members_str = ",".join(str(m) for m in members)
            y_labels.append(f"C{display_idx} (S: {members_str})")

        matrix = np.vstack(proto_rows)
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        # Grades de célula
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, fontsize=8.5)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xticks(np.arange(-0.5, len(feature_names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
        ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = int(matrix[i, j])
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if val == 1 else "#64748b",
                )

        ax.set_title(
            f"Protótipos Aprendidos — $\\rho = {rho}$ ({res['n_active']} classes)",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

    fig.suptitle("Vetores Protótipo (Pesos Top-Down $T$) das Classes Ativas", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "prototipos_por_rho.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_similarity_matrix(X: np.ndarray, situations: np.ndarray, out_dir: Path) -> Path:
    """Mapa de calor da similaridade de Jaccard entre as 10 situações."""
    n = len(situations)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            inter = np.sum((X[i] == 1) & (X[j] == 1))
            union = np.sum((X[i] == 1) | (X[j] == 1))
            sim[i, j] = 1.0 if union == 0 else inter / union

    fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=150)
    im = ax.imshow(sim, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similaridade de Jaccard", fontsize=10)

    labels = [f"S{int(s)}" for s in situations]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_yticklabels(labels, fontsize=9.5)

    for i in range(n):
        for j in range(n):
            val = sim[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if val >= 0.65 else "#0f172a",
                fontweight="bold" if val >= 0.80 and i != j else "normal",
            )

    ax.set_title("Matriz de Similaridade entre as 10 Situações do Processo", fontsize=12.5, fontweight="bold", pad=12)
    fig.tight_layout()
    out_path = out_dir / "matriz_similaridade.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_simulations() -> None:
    base_dir = Path(__file__).resolve().parent
    train_path = base_dir / "treinamento.csv"
    test_path = base_dir / "teste.csv"
    out_dir = base_dir / "graficos"
    out_dir.mkdir(exist_ok=True)

    _, X_train, feature_names = load_dataset(train_path)
    situations_test, X_test, _ = load_dataset(test_path)

    n_samples, n_features = X_train.shape
    vigilance_values = [0.5, 0.8, 0.9, 0.99]

    print("=" * 72)
    print("SIMULAÇÃO DA REDE NEURAL ART-1 — PROCESSO INDUSTRIAL")
    print("=" * 72)

    results = []
    summary_rows = []

    for rho in vigilance_values:
        model = ART1(
            n_features=n_features,
            max_clusters=n_samples,
            vigilance=rho,
            L=2.0,
            max_epochs=100,
        )
        model.fit(X_train)
        predictions = model.predict(X_test)

        clusters: dict[int, list[int]] = {}
        for situation, cluster_idx in zip(situations_test, predictions):
            clusters.setdefault(int(cluster_idx), []).append(int(situation))

        active_clusters = dict(sorted(clusters.items()))
        n_active = len(active_clusters)

        results.append(
            {
                "rho": rho,
                "epochs": model.epochs_trained,
                "n_active": n_active,
                "active_clusters": active_clusters,
                "T": model.T.copy(),
            }
        )

        print(f"\nGrau de vigilância (rho) = {rho}")
        print(f"Épocas até a convergência: {model.epochs_trained}")
        print(f"Quantidade de classes ativas: {n_active}")
        print("-" * 72)

        groups_description = []
        for display_idx, (neuron_idx, members) in enumerate(
            active_clusters.items(), start=1
        ):
            prototype = "".join(str(int(v)) for v in model.T[neuron_idx])
            members_str = ", ".join(f"Situação {s}" for s in members)
            print(
                f"  Classe {display_idx} (Neurônio F2 #{neuron_idx + 1}): "
                f"[{members_str}] | Protótipo: {prototype}"
            )
            groups_description.append(str(members))

        summary_rows.append((rho, n_active, " | ".join(groups_description)))

    print("\n" + "=" * 72)
    print("RESUMO COMPARATIVO")
    print("=" * 72)
    print(f"{'rho':<8} {'Classes Ativas':<16} {'Agrupamentos (Situações)'}")
    print("-" * 72)
    for rho, n_active, desc in summary_rows:
        print(f"{rho:<8} {n_active:<16} {desc}")
    print("=" * 72)

    # Geração dos gráficos
    generated_files = [
        plot_active_classes(results, out_dir),
        plot_cluster_assignments(results, situations_test, out_dir),
        plot_prototypes(results, feature_names, out_dir),
        plot_similarity_matrix(X_test, situations_test, out_dir),
    ]
    print("\nGráficos gerados em:")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    run_simulations()
