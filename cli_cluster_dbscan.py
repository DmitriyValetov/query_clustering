"""
Утилита для кластеризации векторов методом DBSCAN.

Принимает папку с векторами и создает CSV файл с результатами кластеризации.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def load_all_vectors(input_dir: Path) -> Tuple[List[str], np.ndarray]:
    """
    Загружает все векторы из папки с индивидуальными файлами.
    
    Returns:
        Tuple (query_ids, vectors_array)
    """
    print("Загрузка векторов из папки...")
    
    vector_files = sorted(input_dir.glob("*.npz"))
    
    if len(vector_files) == 0:
        raise ValueError(f"Не найдено файлов векторов в папке: {input_dir}")
    
    query_ids = []
    vectors = []
    
    for vector_file in tqdm(vector_files, desc="Загрузка файлов"):
        try:
            data = np.load(vector_file, allow_pickle=True)
            query_id = data["query_id"].item() if isinstance(data["query_id"].item(), str) else str(data["query_id"].item())
            vector = data["vector"]
            
            query_ids.append(query_id)
            vectors.append(vector)
        except Exception as e:
            print(f"\nПредупреждение: ошибка при загрузке {vector_file.name}: {e}")
            continue
    
    if len(vectors) == 0:
        raise ValueError("Не удалось загрузить ни одного вектора")
    
    vectors_array = np.vstack(vectors)
    print(f"Загружено {len(vectors)} векторов, размерность: {vectors_array.shape[1]}")
    
    return query_ids, vectors_array


def estimate_eps_k_distance(vectors: np.ndarray, k: int = 5, metric: str = 'cosine') -> float:
    """
    Оценивает оптимальное значение eps через k-distance graph.
    
    Args:
        vectors: Массив векторов
        k: Количество соседей для анализа
        metric: Метрика расстояния
        
    Returns:
        Предлагаемое значение eps
    """
    print(f"\nОценка eps через k-distance graph (k={k})...")
    
    # Используем меньшую выборку для ускорения на больших датасетах
    sample_size = min(10000, len(vectors))
    if len(vectors) > sample_size:
        print(f"Используется выборка из {sample_size} векторов для оценки eps")
        indices = np.random.choice(len(vectors), sample_size, replace=False)
        sample_vectors = vectors[indices]
    else:
        sample_vectors = vectors
    
    # Вычисляем расстояния до k-го соседа
    if metric == 'cosine':
        nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
    else:
        nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    
    nn.fit(sample_vectors)
    distances, _ = nn.kneighbors(sample_vectors)
    
    # Берем k-й сосед (индекс k, так как 0-й - это сама точка)
    k_distances = distances[:, k]
    k_distances_sorted = np.sort(k_distances)[::-1]
    
    # Используем медиану или перцентиль для оценки eps
    # Обычно берут точку перегиба на графике, но для автоматизации используем перцентиль
    eps_estimate = np.percentile(k_distances_sorted, 75)
    
    print(f"Предлагаемое значение eps: {eps_estimate:.4f}")
    print(f"  (75-й перцентиль k-distance, диапазон: {k_distances_sorted.min():.4f} - {k_distances_sorted.max():.4f})")
    
    return float(eps_estimate)


def cluster_dbscan(
    vectors: np.ndarray,
    eps: float,
    min_samples: int,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Выполняет кластеризацию DBSCAN.
    
    Args:
        vectors: Массив векторов
        eps: Максимальное расстояние между точками в одном кластере
        min_samples: Минимальное количество точек для формирования кластера
        metric: Метрика расстояния
        
    Returns:
        Массив меток кластеров
    """
    print(f"\nКластеризация DBSCAN...")
    print(f"Параметры: eps={eps:.4f}, min_samples={min_samples}, metric={metric}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = dbscan.fit_predict(vectors)
    
    return labels


def calculate_statistics(labels: np.ndarray, eps: float, min_samples: int) -> Dict[str, Any]:
    """
    Вычисляет статистику кластеризации.
    
    Args:
        labels: Массив меток кластеров
        eps: Использованное значение eps
        min_samples: Использованное значение min_samples
        
    Returns:
        Словарь со статистикой
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    n_total = len(labels)
    
    cluster_sizes = {}
    for label in set(labels):
        if label == -1:
            continue
        cluster_sizes[int(label)] = int(np.sum(labels == label))
    
    stats = {
        "total_points": n_total,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_percentage": round(100 * n_noise / n_total, 2) if n_total > 0 else 0,
        "parameters": {
            "eps": eps,
            "min_samples": min_samples
        },
        "cluster_sizes": dict(sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True))
    }
    
    return stats


def save_results(
    query_ids: List[str],
    labels: np.ndarray,
    output_csv: Path,
    stats: Dict[str, Any],
    stats_output: Path
):
    """
    Сохраняет результаты кластеризации.
    
    Args:
        query_ids: Список ID запросов
        labels: Массив меток кластеров
        output_csv: Путь к CSV файлу
        stats: Статистика кластеризации
        stats_output: Путь к JSON файлу со статистикой
    """
    print(f"\nСохранение результатов...")
    
    # Сохраняем CSV
    df = pd.DataFrame({
        'id': query_ids,
        'cluster': labels.astype(int)
    })
    df.to_csv(output_csv, index=False)
    print(f"CSV сохранен: {output_csv}")
    
    # Сохраняем статистику
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Статистика сохранена: {stats_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Кластеризация векторов методом DBSCAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Автоматическое определение eps
  python cli_cluster_dbscan.py -i cache_vectors_reduced -o clusters.csv --auto-eps
  
  # Ручное указание параметров
  python cli_cluster_dbscan.py -i cache_vectors_reduced -o clusters.csv --eps 0.5 --min-samples 10
  
  # С указанием метрики и статистики
  python cli_cluster_dbscan.py -i cache_vectors_reduced -o clusters.csv --eps 0.5 --min-samples 5 --metric euclidean --stats stats.json
        """
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Папка с векторами (индивидуальные .npz файлы)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Путь к выходному CSV файлу с результатами кластеризации"
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Максимальное расстояние между точками в одном кластере (если не указано, будет определено автоматически)"
    )
    
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Минимальное количество точек для формирования кластера (по умолчанию: log(n) или 5)"
    )
    
    parser.add_argument(
        "--auto-eps",
        action="store_true",
        help="Автоматически определить eps через k-distance graph"
    )
    
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Количество соседей для k-distance graph (по умолчанию: 5)"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Метрика расстояния (по умолчанию: cosine)"
    )
    
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Путь к JSON файлу со статистикой (по умолчанию: рядом с CSV файлом)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output)
    
    if not input_dir.exists():
        raise ValueError(f"Входная папка не существует: {input_dir}")
    
    # Определяем путь к файлу статистики
    if args.stats_output:
        stats_output = Path(args.stats_output)
    else:
        stats_output = output_csv.parent / f"{output_csv.stem}_stats.json"
    
    # Загружаем векторы
    query_ids, vectors = load_all_vectors(input_dir)
    
    # Определяем параметры
    n = len(vectors)
    
    # Определяем eps
    if args.eps is not None:
        eps = args.eps
        print(f"\nИспользуется указанное значение eps: {eps}")
    elif args.auto_eps:
        eps = estimate_eps_k_distance(vectors, k=args.k_neighbors, metric=args.metric)
    else:
        # Автоматически определяем, если не указано
        print("\nЗначение eps не указано, выполняется автоматическое определение...")
        eps = estimate_eps_k_distance(vectors, k=args.k_neighbors, metric=args.metric)
    
    # Определяем min_samples
    if args.min_samples is not None:
        min_samples = args.min_samples
    else:
        # Используем эвристику: log(n) или минимум 5
        min_samples = max(5, int(math.log(n)))
        print(f"\nИспользуется автоматически определенное min_samples: {min_samples} (log({n}) = {math.log(n):.2f})")
    
    # Выполняем кластеризацию
    labels = cluster_dbscan(vectors, eps, min_samples, metric=args.metric)
    
    # Вычисляем статистику
    stats = calculate_statistics(labels, eps, min_samples)
    
    # Выводим краткую статистику
    print("\n" + "=" * 60)
    print("Результаты кластеризации:")
    print("=" * 60)
    print(f"Всего точек: {stats['total_points']}")
    print(f"Кластеров: {stats['n_clusters']}")
    print(f"Шум (outliers): {stats['n_noise']} ({stats['noise_percentage']}%)")
    print(f"Параметры: eps={eps:.4f}, min_samples={min_samples}")
    
    # Сохраняем результаты
    save_results(query_ids, labels, output_csv, stats, stats_output)
    
    print("\n" + "=" * 60)
    print("Кластеризация завершена!")
    print("=" * 60)


if __name__ == "__main__":
    main()

