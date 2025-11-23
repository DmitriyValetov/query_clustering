"""
Утилита для распознавания основных тем в кластерах через классификацию запросов.

Для каждого кластера анализирует до 100 запросов и определяет теги/категории.
"""
import os
import json
import re
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from langchain_gigachat.chat_models import GigaChat
import dotenv

from promts import category_classification_prompt_batch

dotenv.load_dotenv()

# Настройки
GIGACHAT_URL = os.environ["GIGACHAT_BASE_URL"]
GIGACHAT_API_PERS = os.environ["GIGACHAT_KEY"]

# Инициализация LLM
llm = GigaChat(
    credentials=GIGACHAT_API_PERS,
    base_url=GIGACHAT_URL + "/v1",
    auth_url=GIGACHAT_URL + "/v1/token",
    verify_ssl_certs=False,
    profanity_check=False,
    model="GigaChat-2-Max",
    timeout=300,
)

# Параметры обработки
BATCH_SIZE = 5  # Запросов в одном батче
PARALLEL_BATCHES = 5  # Количество параллельных батчей
MAX_QUERIES_PER_CLUSTER = 100  # Максимум запросов для анализа на кластер

# Промпт для генерации общего описания тегов кластера
general_tags_prompt_template = """
Ты — эксперт по анализу поисковых запросов. Тебе предоставлен список категорий, которые были определены для запросов из одного кластера.

Твоя задача — создать краткое общее описание тегов/тем этого кластера на основе предоставленных категорий.

Категории кластера:
{categories}

Создай краткое описание (2-4 предложения) на русском языке, которое описывает общую тематику и намерения пользователей в этом кластере.

Верни только описание, без дополнительных пояснений.
""".strip()


def get_query_id(text: str) -> str:
    """Генерирует уникальный ID для запроса (как в других утилитах)"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_clusters(clusters_csv: Path) -> pd.DataFrame:
    """Загружает кластеры из CSV файла"""
    print(f"Загрузка кластеров из {clusters_csv}...")
    df = pd.read_csv(clusters_csv)
    print(f"Загружено {len(df)} записей")
    return df


def load_queries_mapping(csv_files: List[Path]) -> Dict[str, str]:
    """
    Загружает mapping query_id -> query_text из CSV файлов.
    query_id генерируется из текста через хеш (как в других утилитах).
    
    Args:
        csv_files: Список путей к CSV файлам
        
    Returns:
        Словарь {query_id: query_text}
    """
    print("Загрузка запросов из CSV файлов...")
    query_mapping = {}
    
    for csv_file in csv_files:
        print(f"Обработка {csv_file.name}...")
        chunk_size = 100000
        
        for chunk in pd.read_csv(csv_file, usecols=["text", "gigasearch_trigger"], chunksize=chunk_size):
            chunk_filtered = chunk[chunk["gigasearch_trigger"] == 1].copy()
            chunk_filtered = chunk_filtered.dropna(subset=["text"])
            
            for _, row in chunk_filtered.iterrows():
                query_text = str(row["text"]).strip()
                if query_text:
                    query_id = get_query_id(query_text)
                    query_mapping[query_id] = query_text
        
        print(f"  Загружено {len(query_mapping)} уникальных запросов")
    
    return query_mapping


def get_cluster_queries(clusters_df: pd.DataFrame, query_mapping: Dict[str, str], cluster_id: int, max_queries: int) -> List[tuple]:
    """
    Получает список (query_id, query_text) для указанного кластера.
    
    Args:
        clusters_df: DataFrame с кластерами
        query_mapping: Словарь {query_id: query_text}
        cluster_id: ID кластера
        max_queries: Максимальное количество запросов для возврата
    
    Returns:
        Список кортежей (query_id, query_text), максимум max_queries
    """
    cluster_df = clusters_df[clusters_df["cluster"] == cluster_id]
    cluster_queries = []
    
    for _, row in cluster_df.iterrows():
        query_id = str(row["id"])
        if query_id in query_mapping:
            query_text = query_mapping[query_id]
            cluster_queries.append((query_id, query_text))
    
    # Ограничиваем количество
    return cluster_queries[:max_queries]


def classify_batch(queries: List[str]) -> List[str]:
    """Классифицирует пачку запросов одним запросом к API"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
            prompt = category_classification_prompt_batch.format(queries=queries_text)
            
            response = llm.invoke([HumanMessage(prompt)])
            response_text = response.content.strip()
            
            # Парсим JSON ответ
            try:
                if response_text.startswith("```"):
                    response_text = re.sub(r'```(?:json)?\s*', '', response_text)
                    response_text = re.sub(r'```\s*$', '', response_text).strip()
                
                results = json.loads(response_text)
                
                if isinstance(results, list):
                    categories = []
                    for result in results:
                        if isinstance(result, dict) and "category" in result:
                            category = result["category"].strip('"\'.,;:!?')
                            categories.append(category if category else "Неопределено")
                        else:
                            categories.append("Неопределено")
                    
                    if len(categories) < len(queries):
                        while len(categories) < len(queries):
                            categories.append("Неполный ответ")
                    
                    return categories[:len(queries)]
                else:
                    raise ValueError(f"Неверный формат ответа: ожидался список")
                    
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    # Последняя попытка - возвращаем ошибки
                    return ["Ошибка парсинга"] * len(queries)
                import time
                time.sleep(2)
                continue
                
        except Exception as e:
            if attempt == max_retries - 1:
                return ["Ошибка классификации"] * len(queries)
            import time
            time.sleep(2)
    
    return ["Ошибка классификации"] * len(queries)


def generate_general_tags(categories: List[str]) -> str:
    """Генерирует общее описание тегов для кластера"""
    if not categories:
        return "Нет данных для анализа"
    
    # Убираем ошибки и неопределенные категории
    valid_categories = [c for c in categories if c not in ["Ошибка классификации", "Ошибка парсинга", "Неопределено", "Неполный ответ"]]
    
    if not valid_categories:
        return "Не удалось определить категории"
    
    # Формируем список уникальных категорий
    unique_cats = list(set(valid_categories))
    categories_text = "\n".join([f"- {cat}" for cat in unique_cats])
    
    prompt = general_tags_prompt_template.format(categories=categories_text)
    
    try:
        response = llm.invoke([HumanMessage(prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Ошибка при генерации общего описания: {e}")
        return "Не удалось сгенерировать описание"


def process_single_batch(batch_data: tuple) -> tuple:
    """Обрабатывает один батч запросов"""
    batch_num, queries = batch_data
    categories = classify_batch(queries)
    return batch_num, categories


def process_cluster(
    cluster_id: int,
    cluster_queries: List[tuple],
    output_dir: Path
) -> Path:
    """
    Обрабатывает один кластер и сохраняет результаты.
    
    Returns:
        Путь к сохраненному JSON файлу
    """
    if not cluster_queries:
        print(f"Кластер {cluster_id}: нет запросов для обработки")
        return None
    
    print(f"\nОбработка кластера {cluster_id} ({len(cluster_queries)} запросов)...")
    
    # Разбиваем на батчи
    batches = []
    batch_num = 0
    for i in range(0, len(cluster_queries), BATCH_SIZE):
        batch_queries = [q[1] for q in cluster_queries[i:i + BATCH_SIZE]]
        batches.append((batch_num, batch_queries))
        batch_num += 1
    
    # Обрабатываем батчи параллельно
    all_categories = {}
    completed_batches = {}
    
    with ThreadPoolExecutor(max_workers=PARALLEL_BATCHES) as executor:
        future_to_batch = {
            executor.submit(process_single_batch, batch_data): batch_data[0]
            for batch_data in batches
        }
        
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                batch_num_result, categories = future.result()
                completed_batches[batch_num_result] = categories
            except Exception as e:
                print(f"Ошибка при обработке батча {batch_num}: {e}")
                completed_batches[batch_num_result] = ["Ошибка"] * BATCH_SIZE
    
    # Собираем результаты в правильном порядке
    individual_tags = {}
    all_categories_list = []
    
    query_idx = 0
    for batch_num in sorted(completed_batches.keys()):
        categories = completed_batches[batch_num]
        for category in categories:
            if query_idx < len(cluster_queries):
                query_id, query_text = cluster_queries[query_idx]
                individual_tags[query_id] = category
                all_categories_list.append(category)
                query_idx += 1
    
    # Формируем результат
    unique_tags = sorted(list(set(all_categories_list)))
    general_tags = generate_general_tags(all_categories_list)
    
    result = {
        "unique_tags": unique_tags,
        "general_tags": general_tags,
        "individual_tags": individual_tags
    }
    
    # Сохраняем результат
    output_file = output_dir / f"cluster_{cluster_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Кластер {cluster_id}: сохранен в {output_file}")
    print(f"  Уникальных тегов: {len(unique_tags)}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Распознавание основных тем в кластерах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python cli_cluster_themes.py -c clusters.csv -o cluster_themes --csv-files data.csv
  
  python cli_cluster_themes.py -c clusters.csv -o cluster_themes --csv-files data.csv --max-queries 50
        """
    )
    
    parser.add_argument(
        "-c", "--clusters",
        type=str,
        required=True,
        help="Путь к CSV файлу с кластерами (колонки: id, cluster)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Папка для сохранения JSON файлов с темами кластеров"
    )
    
    parser.add_argument(
        "--csv-files",
        type=str,
        nargs='+',
        required=True,
        help="Пути к CSV файлам с запросами (можно указать несколько файлов)"
    )
    
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help=f"Максимум запросов для анализа на кластер (по умолчанию: {MAX_QUERIES_PER_CLUSTER})"
    )
    
    args = parser.parse_args()
    
    clusters_csv = Path(args.clusters)
    output_dir = Path(args.output_dir)
    csv_files = [Path(f) for f in args.csv_files]
    
    if not clusters_csv.exists():
        raise ValueError(f"Файл кластеров не найден: {clusters_csv}")
    
    for csv_file in csv_files:
        if not csv_file.exists():
            raise ValueError(f"CSV файл не найден: {csv_file}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ограничение запросов на кластер
    max_queries_per_cluster = args.max_queries
    
    print("=" * 60)
    print("Распознавание тем в кластерах")
    print("=" * 60)
    
    # Загружаем данные
    clusters_df = load_clusters(clusters_csv)
    query_mapping = load_queries_mapping(csv_files)
    
    # Получаем список уникальных кластеров (исключаем шум -1)
    unique_clusters = sorted([c for c in clusters_df["cluster"].unique() if c != -1])
    
    print(f"\nНайдено {len(unique_clusters)} кластеров для обработки")
    print(f"Параметры: BATCH_SIZE={BATCH_SIZE}, PARALLEL_BATCHES={PARALLEL_BATCHES}, MAX_QUERIES={max_queries_per_cluster}")
    
    # Обрабатываем каждый кластер
    processed_files = []
    for cluster_id in tqdm(unique_clusters, desc="Обработка кластеров"):
        cluster_queries = get_cluster_queries(clusters_df, query_mapping, cluster_id, max_queries_per_cluster)
        if cluster_queries:
            output_file = process_cluster(cluster_id, cluster_queries, output_dir)
            if output_file:
                processed_files.append(output_file)
    
    print("\n" + "=" * 60)
    print("Обработка завершена!")
    print("=" * 60)
    print(f"Обработано кластеров: {len(processed_files)}")
    print(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()

