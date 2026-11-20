from cluster_analyzer import ClusterAnalyzer

INPUT_DATASET = "dataset.csv"
CLUSTERED_DATASET = "dataset_clustered.csv"
MODEL_FILENAME = "kmeans_model.joblib"
REPORT_PREFIX = "cluster_report"


def run_pipeline():
    analyzer = ClusterAnalyzer(k_range=range(2, 16), random_state=42)

    # Шаг 1. Чтение исходной выборки
    table, features = analyzer.load_data(INPUT_DATASET)

    # Шаг 2. Определение оптимального числа кластеров и разбиение объектов
    optimal_k = analyzer.find_optimal_k(features)
    cluster_ids = analyzer.fit_predict(features, n_clusters=optimal_k)

    # Шаг 3. Экспорт размеченной таблицы и сохранение модели K-Means
    table["cluster"] = cluster_ids
    table.to_csv(CLUSTERED_DATASET, index=False, encoding="utf-8")
    print(f"Размеченная выборка сохранена в файл: {CLUSTERED_DATASET}")
    analyzer.save_model(MODEL_FILENAME)

    # Шаг 4. Построение и сохранение графиков
    analyzer.plot_results(features, save_prefix=REPORT_PREFIX)


if __name__ == "__main__":
    run_pipeline()