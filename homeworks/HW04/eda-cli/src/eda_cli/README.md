# EDA CLI Tool

Утилита для быстрого анализа CSV файлов с HTTP-сервисом оценки качества. Делал для домашки, но получилось неплохо.

## Что умеет

### CLI (командная строка)
Две основные команды:
- `overview` - быстро посмотреть что в файле
- `report` - сделать полный отчет с графиками

### HTTP API (FastAPI сервис)
Четыре REST эндпоинта для оценки качества датасетов:
- `GET /health` — health-check сервиса
- `POST /quality` — оценка по агрегированным признакам
- `POST /quality-from-csv` — оценка по CSV-файлу
- `POST /quality-flags-from-csv` — получить полный набор флагов из CSV

## Установка

pip install -e .

## CLI: Как пользоваться

### Быстрый просмотр
eda-cli overview data.csv

Покажет сколько строк/столбцов, типы данных, пропуски.

### Полный отчет
eda-cli report data.csv

Создаст папку `reports/` с:
- Markdown отчетом
- Таблицами в CSV
- Графиками (гистограммы, корреляции, пропуски)

### Настройки отчета

Можно менять параметры:


eda-cli report data.csv
--title "Мой анализ"
--max-hist-columns 10
--top-k-categories 8
--min-missing-share 0.1


**Параметры:**
- `--title` - заголовок отчета
- `--max-hist-columns` - сколько гистограмм строить (по умолчанию 6)
- `--top-k-categories` - сколько топ-значений показывать (по умолчанию 5)
- `--min-missing-share` - порог пропусков для "проблемных" колонок (по умолчанию 0.3)

## HTTP API: Запуск сервера

### Запуск uvicorn


uvicorn eda_cli.api:app --host 0.0.0.0 --port 8000


После запуска:
- API доступен по адресу `http://localhost:8000`
- Интерактивная документация (Swagger UI): `http://localhost:8000/docs`

### Примеры использования API

#### 1. Health-check

curl http://localhost:8000/health


**Ответ:**
json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}


#### 2. Оценка качества по агрегированным признакам

curl -X POST http://localhost:8000/quality \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 5000,
    "n_cols": 25,
    "max_missing_share": 0.15,
    "numeric_cols": 15,
    "categorical_cols": 10
  }'


**Ответ:**
json
{
  "ok_for_model": true,
  "quality_score": 0.85,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 1.23,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 5000,
    "n_cols": 25
  }
}


#### 3. Оценка качества из CSV-файла

curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data.csv"


Эндпоинт:
- Читает CSV-файл
- Запускает EDA-ядро (summarize_dataset, missing_table, compute_quality_flags)
- Возвращает интегральную оценку качества

**Ответ:** как в примере #2

#### 4. Получить полный набор флагов из CSV

curl -X POST http://localhost:8000/quality-flags-from-csv \
  -F "file=@data.csv"


Эндпоинт возвращает **ПОЛНЫЙ набор флагов качества**, включая:
- `has_constant_columns` — найдены константные колонки
- `has_suspicious_id_duplicates` — подозрительные дубликаты в ID-колонках
- Все остальные метрики качества данных

**Ответ:**
json
{
  "flags": {
    "has_constant_columns": false,
    "has_suspicious_id_duplicates": false,
    "quality_score": 0.85,
    "...": "другие флаги и метрики"
  },
  "latency_ms": 45.67,
  "dataset_shape": {
    "n_rows": 5000,
    "n_cols": 25
  }
}


## Что в отчете (CLI режим)

- Описательная статистика
- Проверка качества данных (пропуски, дубликаты и тд)
- Корреляция между числовыми колонками
- Топ категорий для текстовых колонок
- Гистограммы
- Визуализация пропусков

## Примеры использования

### CLI примеры

Обычный запуск:

eda-cli report dataset.csv


Для большого файла (быстро):

eda-cli report big_data.csv --max-hist-columns 3


Строгая проверка качества:

eda-cli report dirty_data.csv --min-missing-share 0.05


### API примеры (Python)

python
import requests

# Прямая оценка по признакам
response = requests.post(
    "http://localhost:8000/quality",
    json={
        "n_rows": 10000,
        "n_cols": 50,
        "max_missing_share": 0.2,
        "numeric_cols": 30,
        "categorical_cols": 20
    }
)
print(response.json())

# Оценка из CSV
with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/quality-from-csv",
        files={"file": f}
    )
print(response.json())

# Получить все флаги
with open("data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/quality-flags-from-csv",
        files={"file": f}
    )
print(response.json())


## Структура проекта


src/eda_cli/
├── __init__.py      # инициализация пакета
├── core.py          # основная логика анализа и проверок качества
├── viz.py           # графики и визуализация
├── cli.py           # командная строка (typer)
└── api.py           # HTTP API (FastAPI) — эндпоинты качества


## Что добавлено

Для домашки добавил:

1. **В `core.py`** — две новые проверки качества:
   - `has_constant_columns` — поиск константных колонок (одно значение для всех строк)
   - `has_suspicious_id_duplicates` — проверка дубликатов в ID-колонках

2. **В `cli.py`** — параметры для настройки отчета:
   - `--title` — заголовок отчета
   - `--max-hist-columns` — количество гистограмм
   - `--top-k-categories` — количество топ-значений
   - `--min-missing-share` — порог пропусков

3. **Новый файл `api.py`** — полноценный HTTP API на FastAPI с четырьмя эндпоинтами:
   - `/health` — health-check
   - `/quality` — оценка по признакам
   - `/quality-from-csv` — оценка из CSV с использованием EDA-ядра
   - `/quality-flags-from-csv` — получить все флаги качества

## Требования

- Python 3.8+
- pandas
- matplotlib
- typer
- fastapi
- uvicorn

## Установка зависимостей


pip install -e .


Или вручную:

pip install pandas matplotlib typer fastapi uvicorn

