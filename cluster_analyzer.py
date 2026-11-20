from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class ClusterAnalyzer:
    """Анализатор для подбора числа кластеров, разбиения выборки и визуализации."""

    def __init__(self, k_range=range(2, 16), random_state=42):
        self.k_values = list(k_range)
        self.seed = random_state
        self.optimal_k = None
        self.kmeans_model = None
        self.cluster_labels = None
        self.summary_table = None

    def load_data(self, csv_path="dataset.csv"):
        """Читает таблицу из CSV и извлекает матрицу числовых признаков."""
        target_path = Path(csv_path)
        if not target_path.is_file():
            raise FileNotFoundError(f"Не удалось найти входной файл: {target_path.resolve()}")

        dataframe = pd.read_csv(target_path)
        feature_names = [col for col in dataframe.columns if col.startswith("feature_")]
        if not feature_names:
            feature_names = dataframe.select_dtypes(include=[np.number]).columns.tolist()

        features = dataframe[feature_names].to_numpy(dtype=np.float64)
        print(f"Данные загружены: {features.shape[0]} строк, {features.shape[1]} признаков ({', '.join(feature_names)})")
        return dataframe, features

    @staticmethod
    def _detect_elbow(k_list, inertia_values):
        """Определяет точку излома (метод локтя) по максимальному расстоянию до хорды."""
        x = np.asarray(k_list, dtype=np.float64)
        y = np.asarray(inertia_values, dtype=np.float64)

        x_norm = (x - x[0]) / (x[-1] - x[0])
        y_norm = (y - y[-1]) / (y[0] - y[-1])

        # Хорда соединяет (0, 1) и (1, 0): прямая x + y - 1 = 0
        distances = (1.0 - x_norm - y_norm) / np.sqrt(2.0)
        best_idx = int(np.argmax(distances))
        return int(x[best_idx]), distances

    def find_optimal_k(self, features):
        """Оценивает качество кластеризации для сетки k и находит точку излома."""
        rows = []
        sub_size = min(10000, len(features))

        for k in self.k_values:
            estimator = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=self.seed,
            )
            preds = estimator.fit_predict(features)
            sil_val = silhouette_score(
                features, preds, sample_size=sub_size, random_state=self.seed
            )
            ch_val = calinski_harabasz_score(features, preds)

            rows.append({
                "k": k,
                "inertia": estimator.inertia_,
                "silhouette": sil_val,
                "calinski_harabasz": ch_val,
            })

        self.summary_table = pd.DataFrame(rows)
        wcss = self.summary_table["inertia"].to_numpy()
        self.summary_table["inertia_drop"] = np.concatenate([[np.nan], -np.diff(wcss)])

        elbow_k, chord_dists = self._detect_elbow(self.k_values, wcss)
        self.summary_table["elbow_score"] = chord_dists
        self.optimal_k = elbow_k

        selected = self.summary_table.loc[self.summary_table["k"] == self.optimal_k].iloc[0]
        print("\nСводная таблица метрик по числу кластеров:")
        print(self.summary_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(
            f"\n>>> Выбрано оптимальное число кластеров (метод локтя): k = {self.optimal_k} "
            f"(Inertia = {selected['inertia']:.2f}, Silhouette = {selected['silhouette']:.4f})"
        )
        return self.optimal_k

    def fit_predict(self, features, n_clusters=None):
        """Обучает финальную модель K-Means на выбранном числе кластеров."""
        target_k = n_clusters if n_clusters is not None else self.optimal_k
        if target_k is None:
            target_k = self.find_optimal_k(features)

        self.optimal_k = int(target_k)
        self.kmeans_model = KMeans(
            n_clusters=self.optimal_k,
            init="k-means++",
            n_init=30,
            random_state=self.seed,
        )
        self.cluster_labels = self.kmeans_model.fit_predict(features)
        return self.cluster_labels

    def save_model(self, output_path="kmeans_model.joblib"):
        """Экспортирует обученный объект KMeans на диск."""
        if self.kmeans_model is None:
            raise RuntimeError("Модель еще не обучена. Сначала вызовите fit_predict().")

        joblib.dump(self.kmeans_model, output_path)
        print(f"Обученная модель сохранена: {output_path}")

    def plot_results(self, features, save_prefix="cluster_report"):
        """Формирует и сохраняет графики подбора k и проекций PCA (2D/3D)."""
        if self.summary_table is None:
            self.find_optimal_k(features)
        if self.kmeans_model is None or self.cluster_labels is None:
            self.fit_predict(features)

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("default")

        # 1. Графики выбора оптимального k (Метод локтя + Коэффициент силуэта)
        fig_metrics, (ax_wcss, ax_sil) = plt.subplots(1, 2, figsize=(13.5, 5.2))

        k_arr = self.summary_table["k"].to_numpy()
        wcss_arr = self.summary_table["inertia"].to_numpy() / 1e6
        sil_arr = self.summary_table["silhouette"].to_numpy()

        ax_wcss.plot(k_arr, wcss_arr, marker="o", lw=2.2, color="#1d4ed8", label="WCSS (Inertia)")
        ax_wcss.plot(
            [k_arr[0], k_arr[-1]],
            [wcss_arr[0], wcss_arr[-1]],
            ls=":",
            lw=1.3,
            color="#64748b",
            label="Хорда (базовая линия)",
        )
        opt_wcss = wcss_arr[k_arr == self.optimal_k][0]
        ax_wcss.scatter([self.optimal_k], [opt_wcss], color="#dc2626", s=95, zorder=6)
        ax_wcss.axvline(
            self.optimal_k,
            color="#dc2626",
            ls="--",
            lw=1.6,
            label=f"Излом (локоть): k = {self.optimal_k}",
        )
        ax_wcss.set_title("Метод локтя (Inertia / WCSS)", fontsize=11.5, fontweight="bold")
        ax_wcss.set_xlabel("Число кластеров (k)")
        ax_wcss.set_ylabel("Сумма квадратов расстояний (×10⁶)")
        ax_wcss.set_xticks(self.k_values)
        ax_wcss.legend(frameon=True)

        ax_sil.plot(k_arr, sil_arr, marker="s", lw=2.2, color="#059669", label="Silhouette Score")
        opt_sil = sil_arr[k_arr == self.optimal_k][0]
        ax_sil.scatter([self.optimal_k], [opt_sil], color="#dc2626", s=95, zorder=6)
        ax_sil.axvline(
            self.optimal_k,
            color="#dc2626",
            ls="--",
            lw=1.6,
            label=f"Выбранное k = {self.optimal_k} ({opt_sil:.4f})",
        )
        ax_sil.set_title("Коэффициент силуэта (Silhouette Score)", fontsize=11.5, fontweight="bold")
        ax_sil.set_xlabel("Число кластеров (k)")
        ax_sil.set_ylabel("Silhouette Score")
        ax_sil.set_xticks(self.k_values)
        ax_sil.legend(frameon=True)

        fig_metrics.suptitle(
            f"Обоснование выбора числа кластеров (оптимум k = {self.optimal_k})",
            fontsize=13,
            fontweight="bold",
        )
        fig_metrics.tight_layout()
        metrics_file = f"{save_prefix}_metrics.png"
        fig_metrics.savefig(metrics_file, dpi=200)

        # 2. Визуализация разбиения в пространстве главных компонент (PCA 2D и 3D)
        n_dims = features.shape[1]
        n_comp = min(3, n_dims)
        pca = PCA(n_components=n_comp, random_state=self.seed)
        coords = pca.fit_transform(features)
        centers_pca = pca.transform(self.kmeans_model.cluster_centers_)
        var_pct = pca.explained_variance_ratio_ * 100.0

        fig_clusters = plt.figure(figsize=(14.5, 6.2))
        palette = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]

        ax_2d = fig_clusters.add_subplot(1, 2, 1)
        for cluster_id in range(self.optimal_k):
            mask = self.cluster_labels == cluster_id
            color = palette[cluster_id % len(palette)]
            ax_2d.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=color,
                label=f"Кластер {cluster_id} (n={mask.sum()})",
                alpha=0.45,
                s=12,
                rasterized=True,
            )
        ax_2d.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c="#111827",
            marker="X",
            s=190,
            edgecolors="white",
            linewidth=1.3,
            label="Центроиды",
            zorder=10,
        )
        ax_2d.set_xlabel(f"ГК 1 ({var_pct[0]:.1f}% дисперсии)")
        ax_2d.set_ylabel(f"ГК 2 ({var_pct[1]:.1f}% дисперсии)")
        ax_2d.set_title(f"Проекция 2D (PCA {n_dims}D → 2D)", fontsize=11.5, fontweight="bold")
        ax_2d.legend(loc="best", fontsize=8.5, frameon=True, markerscale=1.5)

        if n_dims >= 3:
            ax_3d = fig_clusters.add_subplot(1, 2, 2, projection="3d")
            for cluster_id in range(self.optimal_k):
                mask = self.cluster_labels == cluster_id
                color = palette[cluster_id % len(palette)]
                ax_3d.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    coords[mask, 2],
                    c=color,
                    label=f"Кластер {cluster_id}",
                    alpha=0.38,
                    s=9,
                    rasterized=True,
                )
            ax_3d.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                centers_pca[:, 2],
                c="#111827",
                marker="X",
                s=170,
                edgecolors="white",
                linewidth=1.1,
                label="Центроиды",
                zorder=10,
            )
            ax_3d.set_xlabel(f"ГК 1 ({var_pct[0]:.1f}%)")
            ax_3d.set_ylabel(f"ГК 2 ({var_pct[1]:.1f}%)")
            ax_3d.set_zlabel(f"ГК 3 ({var_pct[2]:.1f}%)")
            ax_3d.set_title(
                f"Проекция 3D (PCA {n_dims}D → 3D, Σ={var_pct[:3].sum():.1f}%)",
                fontsize=11.5,
                fontweight="bold",
            )

        fig_clusters.suptitle(
            f"Разбиение выборки dataset.csv на k = {self.optimal_k} кластеров",
            fontsize=13,
            fontweight="bold",
        )
        fig_clusters.tight_layout()
        clusters_file = f"{save_prefix}_clusters.png"
        fig_clusters.savefig(clusters_file, dpi=200)
        print(f"Графические отчеты сохранены: {metrics_file}, {clusters_file}")
        plt.show()