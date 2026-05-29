import os
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

# Конфигурация
DATASET_NAME = "jeggers/competition_math"
# Указываем конкретную конфигурацию 'original', чтобы получить текст задач
CONFIG_NAME = "original"
OUTPUT_DIR = Path("raw/math_dataset")

# Лимит задач на категорию (для теста 100, если хотите всё — поставьте None)
LIMIT_PER_CATEGORY = 100


def format_problem(item):
    """
    Превращает запись датасета в красивый Markdown блок.
    """
    problem = item.get('problem', '')
    solution = item.get('solution', '')
    level = item.get('level', 'Unknown')

    formatted_text = (
        f"## Задача (Level: {level})\n\n"
        f"{problem}\n\n"
        f"### Решение:\n"
        f"{solution}\n\n"
        f"---\n\n"
    )
    return formatted_text


def main():
    print(f"🚀 Начинаю загрузку датасета {DATASET_NAME} (конфигурация: {CONFIG_NAME})...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: добавлен аргумент CONFIG_NAME ---
        dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split='train')
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    print("✅ Датасет скачан. Начинаю обработку...")

    categories = {}

    for item in tqdm(dataset, desc="Processing"):
        category = item.get('type', 'Uncategorized')

        if category not in categories:
            categories[category] = []

        if LIMIT_PER_CATEGORY and len(categories[category]) >= LIMIT_PER_CATEGORY:
            continue

        categories[category].append(item)

    print(f"💾 Сохранение файлов в {OUTPUT_DIR}...")

    if not categories:
        print("⚠️ Категории не найдены.")

    for category, items in categories.items():
        # Очистка имени файла
        safe_name = str(category).replace(" & ", "_").replace(" ", "_").replace("/", "-")
        file_path = OUTPUT_DIR / f"{safe_name}.md"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Сборник задач: {category}\n\n")
            f.write(f"Источник: {DATASET_NAME}\n\n")

            for item in items:
                f.write(format_problem(item))

        print(f"   -> Сохранено: {file_path.name} ({len(items)} задач)")


if __name__ == "__main__":
    main()