from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)
    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)
    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_constant_columns_detection():
    """Проверка обнаружения константных колонок"""
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "status": ["active", "active", "active", "active"],  # Константная!
        "value": [10, 20, 30, 40],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем, что флаг поднялся
    assert flags["has_constant_columns"] is True
    assert "status" in flags["constant_columns_list"]
    assert flags["constant_columns_count"] == 1


def test_no_constant_columns():
    """Проверка что нормальные колонки не считаются константными"""
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "status": ["active", "inactive", "active", "pending"],  # Разные значения
        "value": [10, 20, 30, 40],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Не должно быть константных колонок
    assert flags["has_constant_columns"] is False
    assert flags["constant_columns_count"] == 0


def test_id_duplicates_detection():
    """Проверка обнаружения дубликатов в ID-колонках"""
    df = pd.DataFrame({
        "user_id": [1, 2, 2, 3, 4],  # Дубликат!
        "name": ["Alice", "Bob", "Bob2", "Charlie", "Dave"],
        "value": [10, 20, 30, 40, 50],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем обнаружение дубликатов
    assert flags["has_suspicious_id_duplicates"] is True
    assert len(flags["id_columns_with_duplicates"]) == 1
    
    dup_info = flags["id_columns_with_duplicates"][0]
    assert dup_info["column"] == "user_id"
    assert dup_info["duplicates"] == 1  # 5 значений, 4 уникальных = 1 дубликат


def test_no_id_duplicates():
    """Проверка что уникальные ID не вызывают флаг"""
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],  # Все уникальные
        "product": ["A", "B", "C", "D", "E"],
        "price": [100, 200, 300, 400, 500],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Не должно быть флага о дубликатах
    assert flags["has_suspicious_id_duplicates"] is False
    assert len(flags["id_columns_with_duplicates"]) == 0


def test_quality_score_with_problems():
    """Проверка что quality_score снижается при проблемах"""
    # Датасет с проблемами: константная колонка + дубликаты в ID
    df = pd.DataFrame({
        "product_id": [1, 2, 2, 3],  # Дубликаты
        "constant_col": ["A", "A", "A", "A"],  # Константа
        "value": [10, 20, 30, 40],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Скор должен быть снижен из-за проблем
    assert flags["quality_score"] < 0.8  # Не идеальный
    assert flags["has_constant_columns"] is True
    assert flags["has_suspicious_id_duplicates"] is True


def test_backward_compatibility():
    """Проверка что старый код работает без параметра df"""
    df = _sample_df()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вызов без df (старый способ) должен работать
    flags = compute_quality_flags(summary, missing_df)
    
    assert "quality_score" in flags
    # Новые эвристики не сработают без df, но не должны падать
    assert "has_constant_columns" in flags
    assert "has_suspicious_id_duplicates" in flags