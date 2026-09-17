# V17 — Validation + Multi-View Ensemble

Цель V17 — не добавлять ещё одну сложную модель, а сначала сделать систему, которой можно доверять локально.

## Что мы меняем

1. **Validation по структурам, а не по спектрам.**
   Одна и та же структура может иметь несколько MS/MS-спектров. Если часть её спектров попадёт в train, а часть в validation, метрика будет завышена. В текущем notebook стабильный structural key — `inchikey14`, поэтому split делаем по нему.

2. **Pseudo Class-2 validation.**
   Для validation-структуры убираем её собственные спектры из spectral library, но оставляем структуру среди кандидатов. Это имитирует ситуацию «структура известна в базе, но эталонного спектра нет».

3. **Несколько views одного спектра.**
   Вместо одного preprocessing считаем несколько вариантов:
   - low-noise: intensity floor 0.2%
   - balanced: 1%
   - strict: 2%
   - optional: top peaks per m/z window

4. **Rank Fusion вместо усреднения raw score.**
   Разные методы имеют разные шкалы score. Мы сначала ранжируем кандидатов внутри каждого метода, затем объединяем ранги через Reciprocal Rank Fusion (RRF).

5. **Сначала candidate recall, потом ranker.**
   Если правильной структуры нет в Top-100 candidate pool, никакой ML-ranker её не восстановит. Поэтому для каждой версии считаем:
   - Recall@25
   - Recall@50
   - Recall@100
   - MRR@25

6. **`sample_submission.csv` — это контракт, а не обучающие данные.**
   Он задаёт колонки, список `molecule_id`, порядок строк и формат 25 SMILES через `;`. Правильных ответов в нём нет, поэтому качество модели мы улучшаем на pseudo-validation из `train.parquet`, а sample используем только для безопасной сборки финального CSV.

---

## Минимальный план запуска

### Шаг 1. Сделать pseudo Class-2 validation

Для текущего train используем `inchikey14`, потому что именно этот ключ использует spectral library/candidate pool notebook-а.

```python
from src.validation import make_pseudo_class2_split

library_df, val_df = make_pseudo_class2_split(
    train_df,
    key_col='inchikey14',
    val_fraction=0.20,
    random_state=42,
)
```

Что произошло:

- 80% уникальных структур остаются в spectral library;
- 20% структур становятся query-примерами;
- **ни одного спектра этих 20% нет в library**;
- сами структуры можно оставить в candidate database — это и есть pseudo Class 2.

### Шаг 2. Проверить отсутствие leakage

```python
from src.validation import assert_no_group_leakage

assert_no_group_leakage(
    library_df,
    val_df,
    group_col='inchikey14',
)
```

Если assertion проходит, одна и та же структура не присутствует по обе стороны split.

### Шаг 3. Получить несколько ranking lists

Не нужно полностью запускать pipeline четыре раза. Тяжёлые candidate features лучше посчитать один раз, а затем получить несколько score-векторов.

Первый набор экспериментов:

```python
views = {
    'entropy_002': scores_entropy_floor_002,
    'entropy_010': scores_entropy_floor_010,
    'entropy_020': scores_entropy_floor_020,
    'analog': scores_analog,
    'metfrag': scores_metfrag,
}
```

Это не обязательные финальные веса/параметры. Их задача — проверить, действительно ли разные способы обработки ошибаются по-разному.

### Шаг 4. Объединить результаты

Если все views используют один candidate pool:

```python
from src.rank_fusion import reciprocal_rank_fusion

final_scores = reciprocal_rank_fusion(
    candidate_ids,
    score_views=views,
    weights={
        'entropy_002': 1.0,
        'entropy_010': 1.1,
        'entropy_020': 0.8,
        'analog': 1.2,
        'metfrag': 0.9,
    },
    k=20,
)
```

Если это действительно независимые прогоны с разными candidate sets (например 5 ppm и 20 ppm), используем списки напрямую:

```python
from src.rank_fusion import fused_ranking_from_lists

ranking = fused_ranking_from_lists(
    {
        'ppm_5': ranking_5ppm,
        'ppm_10': ranking_10ppm,
        'ppm_20': ranking_20ppm,
    },
    weights={'ppm_5': 1.1, 'ppm_10': 1.0, 'ppm_20': 0.8},
    k=20,
)
```

RRF не складывает raw scores. Для каждого view кандидат получает очки по позиции:

`weight / (k + rank)`

### Шаг 5. Проверить метрики

Для локального benchmark удобнее оценивать candidate keys (`inchikey14`), а не строку SMILES — так разные текстовые представления одной структуры не создают ложную ошибку.

```python
from src.validation import evaluate_ranking, identity_truth

truth = identity_truth(val_df, key_col='inchikey14')
metrics = evaluate_ranking(predictions, truth)
print(metrics)
```

Где `predictions` имеет вид:

```python
{
    query_inchikey14: [candidate_key_1, candidate_key_2, ...]
}
```

### Шаг 6. Собрать submission строго по sample-файлу

Текущий sample-файл имеет две колонки: `molecule_id` и `smiles`. Поле `smiles` содержит 25 кандидатов, разделённых `;`. Не нужно вручную собирать DataFrame и надеяться, что порядок совпадёт.

```python
import pandas as pd
from src.submission import build_submission, validate_submission

sample = pd.read_csv(SAMPLE)

# predictions_for_test: molecule_id -> SMILES от лучшего к худшему
submission = build_submission(
    sample_submission=sample,
    predictions=predictions_for_test,
    topk=25,
    fallback='CCO',
)

validate_submission(submission, sample, topk=25)
submission.to_csv('submission.csv', index=False)
```

`build_submission()`:

- сохраняет **точно тот же порядок molecule_id**, что и sample;
- убирает дубликаты SMILES в начале списка, чтобы не тратить ranking slots;
- дополняет список до 25 элементов, если кандидатов меньше;
- падает с понятной ошибкой, если для какого-то molecule_id prediction отсутствует;
- не хардкодит число test-молекул.

---

## Как мы улучшаем алгоритм «на основе примеров»

`sample_submission.csv` не содержит правильных ответов. Поэтому реальные примеры для обучения/сравнения берём из `train.parquet` и сами превращаем часть train в скрытый экзамен.

Для каждой идеи сравниваем **один и тот же validation split**:

| Experiment | Что меняем | Зачем |
|---|---|---|
| E0 | текущий V16 | контрольная точка |
| E1 | intensity floor 0.2% → 1% | проверить влияние слабых шумовых peaks |
| E2 | intensity floor 2% | ещё более строгая очистка |
| E3 | 5 / 10 / 20 ppm rankings + RRF | твоя идея нескольких прогонов |
| E4 | direct fragment + neutral-loss ranking | получить другой тип ошибок |
| E5 | E1–E4 fusion | проверить реальную пользу ensemble |

Для каждого эксперимента сохраняем:

```text
Recall@25
Recall@50
Recall@100
MRR@25
runtime
Top-10 overlap с baseline
```

Меняем **одну вещь за раз**. Если E1 хуже E0, выкидываем E1. Если E3 даёт тот же Top-10 почти везде — multi-ppm ensemble не нужен. Если Recall@100 хороший, а MRR плохой — тогда уже улучшаем ranker.

---

## Как понимать результаты

### Сценарий A

Recall@100 низкий, например 0.55.

**Вывод:** проблема не в ranker. Правильной структуры часто нет среди кандидатов. Следующий шаг — расширять candidate generation / formula filtering / PubChem subset.

### Сценарий B

Recall@100 высокий, например 0.90, но MRR@25 низкий.

**Вывод:** candidates хорошие, но порядок плохой. Тогда имеет смысл LambdaRank / pairwise ranking / новые features.

### Сценарий C

Каждый отдельный view даёт похожий MRR, но ошибается на разных молекулах.

**Вывод:** ensemble имеет смысл.

### Сценарий D

Все views дают почти одинаковый Top-10.

**Вывод:** ensemble почти ничего не даст. Нужен действительно другой источник сигнала, а не другой коэффициент того же cosine.

---

## Что НЕ делать пока

Не добавлять сразу одновременно DreaMS, PubChem, SIRIUS, LambdaRank и graph generation. Иначе если score изменится, мы не поймём почему.

Правильная последовательность:

1. pseudo-validation
2. candidate recall
3. multi-view preprocessing
4. independent-run rank fusion
5. sample-submission validation
6. ranker upgrade
7. formula gate
8. external candidate expansion
9. graph-edit generation

---

## Definition of Done для V17

V17 считается готовой, если:

- validation split не содержит одну и ту же `inchikey14` в library и validation;
- есть pseudo Class-2 evaluation;
- считаются Recall@25/50/100 и MRR@25;
- есть минимум 3 score views или независимых rankings;
- есть RRF ensemble;
- submission собирается через sample-файл и проходит format validation;
- все параметры и результаты записываются в таблицу экспериментов;
- public Kaggle submission выполняется только после локального улучшения.
