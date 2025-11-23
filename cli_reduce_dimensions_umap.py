"""
Утилита для снижения размерности векторов методом UMAP.

Принимает папку с векторами и создает сжатые версии векторов в указанной папке.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import umap


def load_all_vectors(input_dir: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Загружает все векторы из папки с индивидуальными файлами.
    
    Returns:
        Tuple (query_ids, query_texts, vectors_array)
    """
    print("Загрузка векторов из папки...")
    
    vector_files = sorted(input_dir.glob("*.npz"))
    
    if len(vector_files) == 0:
        raise ValueError(f"Не найдено файлов векторов в папке: {input_dir}")
    
    query_ids = []
    query_texts = []
    vectors = []
    
    for vector_file in tqdm(vector_files, desc="Загрузка файлов"):
        try:
            data = np.load(vector_file, allow_pickle=True)
            query_id = data["query_id"].item() if isinstance(data["query_id"].item(), str) else str(data["query_id"].item())
            query_text = data["query_text"].item() if isinstance(data["query_text"].item(), str) else str(data["query_text"].item())
            vector = data["vector"]
            
            query_ids.append(query_id)
            query_texts.append(query_text)
            vectors.append(vector)
        except Exception as e:
            print(f"\nПредупреждение: ошибка при загрузке {vector_file.name}: {e}")
            continue
    
    if len(vectors) == 0:
        raise ValueError("Не удалось загрузить ни одного вектора")
    
    vectors_array = np.vstack(vectors)
    print(f"Загружено {len(vectors)} векторов, размерность: {vectors_array.shape[1]}")
    
    return query_ids, query_texts, vectors_array


def train_umap(vectors: np.ndarray, n_components: int) -> umap.UMAP:
    """
    Обучает модель UMAP на всех векторах.
    
    Args:
        vectors: Массив всех векторов
        n_components: Целевая размерность
        
    Returns:
        Обученная модель UMAP
    """
    print(f"\nОбучение UMAP для снижения размерности до {n_components}...")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    
    reducer.fit(vectors)
    print(f"Обучение завершено. Размерность будет снижена: {vectors.shape[1]} -> {n_components}")
    
    return reducer


def save_reduced_vector(
    query_id: str,
    query_text: str,
    reduced_vector: np.ndarray,
    output_dir: Path
):
    """Сохраняет сжатый вектор в отдельный файл"""
    vector_file = output_dir / f"{query_id}.npz"
    np.savez_compressed(
        vector_file,
        query_id=query_id,
        query_text=query_text,
        vector=reduced_vector
    )


def reduce_dimensions(
    input_dir: Path,
    output_dir: Path,
    n_components: int
):
    """
    Снижает размерность векторов методом UMAP.
    
    Args:
        input_dir: Папка с исходными векторами
        output_dir: Папка для сохранения сжатых векторов
        n_components: Целевая размерность векторов
    """
    print("=" * 60)
    print("Снижение размерности векторов методом UMAP")
    print("=" * 60)
    
    # Создаем выходную папку
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем все векторы
    query_ids, query_texts, vectors = load_all_vectors(input_dir)
    
    # Обучаем UMAP
    reducer = train_umap(vectors, n_components)
    
    # Применяем трансформацию ко всем векторам
    print(f"\nПрименение трансформации к векторам...")
    reduced_vectors = reducer.transform(vectors)
    
    # Сохраняем каждый сжатый вектор
    print(f"\nСохранение сжатых векторов в {output_dir}...")
    for i, query_id in enumerate(tqdm(query_ids, desc="Сохранение векторов")):
        save_reduced_vector(
            query_id,
            query_texts[i],
            reduced_vectors[i],
            output_dir
        )
    
    print("\n" + "=" * 60)
    print("Снижение размерности завершено!")
    print("=" * 60)
    print(f"Обработано векторов: {len(query_ids)}")
    print(f"Исходная размерность: {vectors.shape[1]}")
    print(f"Новая размерность: {n_components}")
    print(f"\nСжатые векторы сохранены в: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Снижение размерности векторов методом UMAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Снижение размерности до 128
  python cli_reduce_dimensions_umap.py -i cache_vectors_unpacked -o cache_vectors_reduced -d 128
  
  # Снижение размерности до 64
  python cli_reduce_dimensions_umap.py -i cache_vectors_unpacked -o cache_vectors_reduced -d 64
        """
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Папка с исходными векторами (индивидуальные .npz файлы)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Папка для сохранения сжатых векторов"
    )
    
    parser.add_argument(
        "-d", "--dimensions",
        type=int,
        required=True,
        help="Размерность итоговых векторов"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Входная папка не существует: {input_dir}")
    
    if args.dimensions <= 0:
        raise ValueError("Размерность должна быть положительным числом")
    
    reduce_dimensions(
        input_dir=input_dir,
        output_dir=output_dir,
        n_components=args.dimensions
    )


if __name__ == "__main__":
    main()

