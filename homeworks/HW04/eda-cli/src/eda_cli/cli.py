from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6, 
        "--max-hist-columns",
        help="Максимум числовых колонок для гистограмм."
    ),
    top_k_categories: int = typer.Option(
        5,
        "--top-k-categories",
        help="Количество top-значений для категориальных признаков."
    ),
    title: str = typer.Option(
        "EDA-отчёт",
        "--title",
        help="Заголовок отчёта в Markdown."
    ),
    min_missing_share: float = typer.Option(
        0.3,
        "--min-missing-share",
        help="Порог доли пропусков (0.0-1.0), выше которого колонка считается проблемной."
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    # Валидация параметров
    if not 0.0 <= min_missing_share <= 1.0:
        raise typer.BadParameter("min_missing_share должен быть в диапазоне [0.0, 1.0]")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)

    # Используем top_k_categories параметр
    top_cats = top_categories(df, max_columns=5, top_k=top_k_categories)

    # 2. Качество в целом (передаем df для новых эвристик)
    quality_flags = compute_quality_flags(summary, missing_df, df)

    # 3. Находим проблемные колонки по порогу пропусков
    problematic_columns = []
    if not missing_df.empty:
        problematic_columns = missing_df[
            missing_df['missing_share'] > min_missing_share
        ].index.tolist()

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        # Используем кастомный заголовок
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        # Параметры отчёта
        f.write("## Параметры отчёта\n\n")
        f.write(f"- Максимум гистограмм: **{max_hist_columns}**\n")
        f.write(f"- Top-K категорий: **{top_k_categories}**\n")
        f.write(f"- Порог проблемных пропусков: **{min_missing_share:.1%}**\n\n")

        # Качество данных
        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")

        # Новые эвристики
        if quality_flags.get('has_constant_columns'):
            f.write(f"- Константные колонки: **{quality_flags['constant_columns_list']}**\n")

        if quality_flags.get('has_suspicious_id_duplicates'):
            f.write(f"- Дубликаты в ID-колонках:\n")
            for item in quality_flags.get('id_columns_with_duplicates', []):
                f.write(f"  - `{item['column']}`: {item['duplicates']} дубликатов\n")

        f.write("\n")

        # Колонки
        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        # Пропуски с информацией о проблемных колонках
        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

            if problematic_columns:
                f.write(f"### Проблемные колонки (пропусков > {min_missing_share:.1%})\n\n")
                for col in problematic_columns:
                    share = missing_df.loc[col, 'missing_share']
                    count = int(missing_df.loc[col, 'missing_count'])
                    f.write(f"- `{col}`: {count} пропусков ({share:.1%})\n")
                f.write("\n")

        # Корреляция
        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        # Категориальные признаки с информацией о Top-K
        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"Показаны top-{top_k_categories} значений для каждого признака.\n\n")
            f.write("См. файлы в папке `top_categories/`.\n\n")

        # Гистограммы
        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"Построены гистограммы для первых {max_hist_columns} числовых колонок.\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 6. Картинки (используем max_hist_columns)
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"  Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"   Заголовок: '{title}'")
    typer.echo(f"\n  Основной файл:")
    typer.echo(f"   - {md_path}")
    typer.echo(f"\n  Табличные файлы:")
    typer.echo(f"   - summary.csv, missing.csv, correlation.csv")
    typer.echo(f"   - top_categories/*.csv (top-{top_k_categories} значений)")
    typer.echo(f"\n  Графики:")
    typer.echo(f"   - hist_*.png (макс. {max_hist_columns} колонок)")
    typer.echo(f"   - missing_matrix.png, correlation_heatmap.png")

    if problematic_columns:
        typer.echo(f"\n Проблемные колонки (пропусков > {min_missing_share:.1%}):")
        for col in problematic_columns:
            typer.echo(f"   - {col}")


if __name__ == "__main__":
    app()
