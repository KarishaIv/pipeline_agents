#!/usr/bin/env python3
"""
Скрипт для загрузки модели эмбеддингов multilingual-e5-small
для sentence-transformers.

Запуск:
    python download_model.py

После успешной загрузки модель будет закэширована и main.py
будет работать быстро (модель берется из кэша).
"""

import os
import shutil
import sys
from pathlib import Path


def clear_model_cache(model_name: str = "intfloat/multilingual-e5-small"):
    """Очищает старый кэш модели, если он есть."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    # Ищем папки с моделью
    for item in cache_dir.iterdir():
        if item.is_dir() and "multilingual-e5-small" in item.name:
            print(f"Удаляем старый кэш: {item}")
            shutil.rmtree(item)
            return True
    return False


def download_model():
    """Загружает модель для sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model_name = "intfloat/multilingual-e5-small"
    print("Загрузка модели эмбеддингов")
    print(f"Модель: {model_name}")
    print()

    # Очищаем старый кэш (если есть)
    if clear_model_cache(model_name):
        print(" Старый кэш удален")
        print()

    try:
        # Загружаем модель с явным разрешением на загрузку кода
        print("Загрузка модели...")
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            local_files_only=False
        )

        # Тестируем модель
        print("Тестирование модели...")
        test_texts = ["Тестовый запрос", "Еще один пример"]
        embeddings = model.encode(test_texts, show_progress_bar=False)

        dim = model.get_sentence_embedding_dimension()

        print()
        print("Модель успешно загружена")
        print(f"Размер эмбеддинга: {dim} dim")
        print(f"Пример вектора (первые 5 значений): {embeddings[0][:5]}")
        print(f"Путь к кэшу: {Path.home() / '.cache' / 'huggingface' / 'hub'}")
        print()
        print("Теперь можно запускать main.py")

        return True

    except Exception as e:
        print()
        print("Ошибка при загрузке модели")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {e}")
        return False


def check_model_exists():
    """Проверяет, есть ли модель в кэше."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if "multilingual-e5-small" in str(repo.repo_id):
            print(f"Модель найдена в кэше: {repo.repo_id}")
            for rev in repo.revisions:
                print(f"   • {rev.commit_hash[:8]}: {rev.size_on_disk_str}")
            return True
    return False


def main():
    """Точка входа."""
    print("Проверка наличия модели в кэше...")
    if check_model_exists():
        print()
        choice = input("Модель уже есть в кэше. Загрузить заново? (y/n): ")
        if choice.lower() != "y":
            print("Готово! Можно запускать main.py")
            return 0

    print()
    success = download_model()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())