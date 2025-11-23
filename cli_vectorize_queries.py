"""
Скрипт для векторизации поисковых запросов и сохранения в numpy файл.
Векторы сохраняются локально в указанную папку для последующей кластеризации.
"""
import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain_gigachat.embeddings import GigaChatEmbeddings
import dotenv

dotenv.load_dotenv()

# Настройки
GIGACHAT_URL = os.environ["GIGACHAT_BASE_URL"]
GIGACHAT_API_PERS = os.environ["GIGACHAT_KEY"]

# Инициализация модели для эмбеддингов
embeddings = GigaChatEmbeddings(
    credentials=GIGACHAT_API_PERS,
    base_url=GIGACHAT_URL + "/v1",
    auth_url=GIGACHAT_URL + "/v1/token",
    verify_ssl_certs=False,
    model="EmbeddingsGigaR"
)


def get_query_id(text: str) -> str:
    """Генерирует уникальный ID для запроса"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_queries_chunks(input_files: List[str], chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """Генератор для постепенной загрузки запросов из CSV файлов"""
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Предупреждение: файл {file_path} не найден, пропускаю")
            continue
            
        print(f"Обработка {file_path}...")
        
        for chunk in pd.read_csv(file_path, usecols=["text", "gigasearch_trigger"], chunksize=chunk_size):
            chunk_filtered = chunk[chunk["gigasearch_trigger"] == 1].copy()
            chunk_filtered = chunk_filtered.dropna(subset=["text"])
            if not chunk_filtered.empty:
                chunk_filtered["query_id"] = chunk_filtered["text"].apply(get_query_id)
                yield chunk_filtered


def load_vectors_metadata(metadata_file: Path) -> Dict[str, Any]:
    """Загружает метаданные о сохраненных векторах"""
    if not metadata_file.exists():
        return {"total_vectors": 0, "query_ids": set()}
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        metadata["query_ids"] = set(metadata["query_ids"])  # Преобразуем обратно в set
        return metadata


def save_vectors_metadata(metadata: Dict[str, Any], metadata_file: Path):
    """Сохраняет метаданные о векторах"""
    metadata_copy = metadata.copy()
    metadata_copy["query_ids"] = list(metadata_copy["query_ids"])  # Преобразуем set в list для JSON
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_copy, f, ensure_ascii=False, indent=2)


def vectorize_single_query(query_data: Tuple[str, str]) -> Tuple[str, str, List[float]]:
    """Векторизует один запрос"""
    query_id, query_text = query_data
    try:
        vector = embeddings.embed_query(query_text)
        return query_id, query_text, vector
    except Exception as e:
        print(f"Ошибка при векторизации запроса {query_id}: {e}")
        return None, None, None


def save_single_vector_to_file(query_id: str, query_text: str, vector: List[float], vectors_dir: Path):
    """Сохраняет один вектор в отдельный файл с именем query_id"""
    vector_file = vectors_dir / f"{query_id}.npz"
    np.savez_compressed(
        vector_file,
        query_id=query_id,
        query_text=query_text,
        vector=np.array(vector)
    )
    return vector_file


def vectorize_and_save_to_numpy(
    input_files: List[str],
    output_dir: Path,
    parallel_workers: int = 5,
    chunk_size: int = 10000,
    limit: int = None
):
    """Векторизует запросы постепенно и сохраняет каждый вектор в отдельный файл"""
    print("Постепенная векторизация запросов...")
    if limit:
        print(f"Ограничение: обрабатываем максимум {limit} новых запросов")
    print("Сохранение каждого вектора в отдельный файл")
    print(f"Входные файлы: {', '.join(input_files)}")
    print(f"Выходная папка: {output_dir}")
    
    # Создаем выходную папку если её нет
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Файл метаданных в той же папке
    metadata_file = output_dir / "metadata.json"
    
    # Загружаем метаданные о уже сохраненных векторах
    metadata = load_vectors_metadata(metadata_file)
    existing_ids = metadata["query_ids"]
    
    # Счетчики
    total_processed = 0
    total_new = 0
    
    # Обрабатываем данные по частям
    with tqdm(desc="Обработка запросов", total=limit) as pbar:
        for chunk_df in load_queries_chunks(input_files, chunk_size=chunk_size):
            # Проверяем лимит
            if limit and total_new >= limit:
                print(f"\nДостигнут лимит в {limit} запросов")
                break
            
            # Фильтруем уже обработанные запросы
            chunk_df_new = chunk_df[~chunk_df["query_id"].isin(existing_ids)].copy()
            
            if chunk_df_new.empty:
                total_processed += len(chunk_df)
                if not limit:
                    pbar.update(len(chunk_df))
                continue
            
            # Ограничиваем количество запросов из чанка если нужно
            if limit and total_new + len(chunk_df_new) > limit:
                remaining = limit - total_new
                chunk_df_new = chunk_df_new.head(remaining)
            
            # Подготавливаем данные для параллельной векторизации
            query_data_list = [
                (row["query_id"], row["text"])
                for _, row in chunk_df_new.iterrows()
            ]
            
            # Векторизуем параллельно
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(vectorize_single_query, query_data): query_data
                    for query_data in query_data_list
                }
                
                for future in as_completed(futures):
                    # Проверяем лимит перед обработкой результата
                    if limit and total_new >= limit:
                        # Отменяем оставшиеся задачи
                        for f in futures:
                            f.cancel()
                        break
                    
                    query_id, query_text, vector = future.result()
                    
                    if vector is not None:
                        # Сохраняем вектор сразу в отдельный файл
                        save_single_vector_to_file(query_id, query_text, vector, output_dir)
                        
                        existing_ids.add(query_id)  # Обновляем кэш
                        metadata["query_ids"].add(query_id)
                        metadata["total_vectors"] += 1
                        total_new += 1
                        
                        # Периодически сохраняем метаданные (каждые 100 векторов для производительности)
                        if total_new % 100 == 0:
                            save_vectors_metadata(metadata, metadata_file)
                    
                    total_processed += 1
                    pbar.update(1)
                    
                    # Проверяем лимит после обработки
                    if limit and total_new >= limit:
                        break
            
            # Проверяем лимит после обработки чанка
            if limit and total_new >= limit:
                break
    
    # Финальное сохранение метаданных
    save_vectors_metadata(metadata, metadata_file)
    
    # Финальная статистика
    final_metadata = load_vectors_metadata(metadata_file)
    print(f"\nОбработано запросов: {total_processed}")
    print(f"Новых запросов векторизовано: {total_new}")
    print(f"Всего векторов сохранено: {final_metadata['total_vectors']}")
    
    if final_metadata["total_vectors"] > 0:
        # Загружаем первый вектор чтобы узнать размерность
        first_query_id = list(final_metadata["query_ids"])[0]
        first_vector_file = output_dir / f"{first_query_id}.npz"
        if first_vector_file.exists():
            first_vector_data = np.load(first_vector_file, allow_pickle=True)
            return len(first_vector_data["vector"])
    
    # Получаем размерность из тестового запроса если нет данных
    sample_vector = embeddings.embed_query("тест")
    return len(sample_vector)


def main():
    parser = argparse.ArgumentParser(
        description="Векторизация поисковых запросов и сохранение в отдельные файлы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Использование файлов по умолчанию
  python vectorize_queries_cli.py -o cache_vectors
  
  # Указание конкретных файлов
  python vectorize_queries_cli.py -i data.csv -o cache_vectors
  
  # С ограничением количества и настройкой параллельных потоков
  python vectorize_queries_cli.py -i data.csv -o vectors -l 1000 -w 10
  
  # Полная настройка всех параметров
  python vectorize_queries_cli.py -i file1.csv file2.csv -o output_dir -w 8 -c 5000 -l 500
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        default=["data.csv"],
        help="Входные CSV файлы для обработки (по умолчанию: data.csv)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="cache_vectors",
        help="Папка для сохранения векторов (по умолчанию: cache_vectors)"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=5,
        help="Количество параллельных потоков (воркеров) для векторизации (по умолчанию: 5)"
    )
    
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=10000,
        help="Размер чанка для чтения из CSV (по умолчанию: 10000)"
    )
    
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Ограничение количества новых запросов для обработки (по умолчанию: без ограничений)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Векторизация поисковых запросов")
    print("=" * 60)
    
    # Преобразуем пути
    input_files = [str(f) for f in args.input]
    output_dir = Path(args.output)
    
    # Векторизация и сохранение в numpy файл
    vector_size = vectorize_and_save_to_numpy(
        input_files=input_files,
        output_dir=output_dir,
        parallel_workers=args.workers,
        chunk_size=args.chunk_size,
        limit=args.limit
    )
    
    print("\n" + "=" * 60)
    print("Векторизация завершена!")
    print("=" * 60)
    print(f"\nВекторы сохранены в: {output_dir}")
    print(f"Метаданные: {output_dir / 'metadata.json'}")
    print(f"Размерность векторов: {vector_size}")
    print(f"\nДля кластеризации запустите: python cluster_queries.py")


if __name__ == "__main__":
    main()

