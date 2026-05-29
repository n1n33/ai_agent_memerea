import statistics
from datetime import datetime
import os


def generate_markdown_report(results, total_time, output_file, model_name):
    """
    Формирует официальный отчет (сводка + ошибки).
    (Код этой функции остается без изменений, как в вашем примере)
    """
    if not results:
        print("[WARNING] Данные для формирования отчета отсутствуют.")
        return

    scores = [r['score'] for r in results]
    latencies = [r['latency'] for r in results]
    avg_score = statistics.mean(scores) if scores else 0

    report = f"""# Протокол тестирования аналитической системы

**Дата формирования:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Наименование системы:** {model_name}
**Объем выборки:** {len(results)} тестовых сценариев

## 1. Сводные показатели эффективности
| Показатель | Значение | Комментарий |
|------------|----------|-------------|
| **Средняя оценка качества** | **{avg_score:.2f} / 5.00** | Целевой показатель > 4.0 |
| **Среднее время отклика** | {statistics.mean(latencies):.2f} с | Latency |
| **Медианное время отклика** | {statistics.median(latencies):.2f} с | Устойчивость системы |
| **Общее время выполнения** | {total_time:.1f} с | |

## 2. Распределение результатов
* **Высокое качество (5):** {scores.count(5)}
* **Удовлетворительное качество (3-4):** {scores.count(4) + scores.count(3)}
* **Низкое качество (1-2):** {scores.count(2) + scores.count(1)}
* **Отказ системы (0):** {scores.count(0)}

## 3. Детализация несоответствий (Оценка <= 3)
Ниже приведен список сценариев, требующих внимания разработчика.
"""
    for r in results:
        if r['score'] <= 3:
            report += f"""
### Сценарий: {r['question']}
* **Категория:** {r.get('category', 'N/A')}
* **Оценка системы:** {r['score']}
* **Фактический ответ:** {r['prediction']}
* **Время обработки:** {r['latency']:.2f} с
---
"""
    try:
        abs_path = os.path.abspath(output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[INFO] Отчет успешно сохранен в файл: {abs_path}")
    except IOError as e:
        print(f"[ERROR] Ошибка записи файла отчета: {e}")


# --- НОВАЯ ФУНКЦИЯ ДЛЯ ЛОГОВ ---
def save_detailed_logs(results, output_file):
    """
    Сохраняет полный лог всех запросов и ответов (Q&A) в отдельный файл.
    """
    if not results:
        return

    log_content = f"# Полный лог тестирования (Q&A)\n"
    log_content += f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    log_content += f"**Всего запросов:** {len(results)}\n\n"
    log_content += "---\n"

    for i, row in enumerate(results, 1):

        log_content += f"## {i}. {row['question']}\n\n"
        log_content += f"- **Категория:** {row['category']}\n"
        log_content += f"(Оценка: {row['score']})\n"
        log_content += f"- **Время:** {row['latency']:.2f} сек\n\n"
        log_content += f"### Ответ системы:\n"
        # Используем цитирование для ответа, чтобы отделить его визуально
        log_content += f"> {row['prediction'].replace(chr(10), chr(10) + '> ')}\n\n"
        log_content += "---\n\n"

    try:
        abs_path = os.path.abspath(output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(log_content)
        print(f"[INFO] Полные логи сохранены в файл: {abs_path}")
    except IOError as e:
        print(f"[ERROR] Ошибка записи лог-файла: {e}")