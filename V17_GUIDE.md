# V17 — Validation + Multi-View Ensemble

Цель V17 — не добавлять ещё одну сложную модель, а сначала сделать систему, которой можно доверять локально.

## Что мы меняем

1. **Validation по молекулам, а не по спектрам.**
   Одна молекула может иметь несколько MS/MS-спектров. Если часть её спектров попадёт в train, а часть в validation, метрика будет завышена. Поэтому разделяем данные по molecule key / InChIKey.

2. **Pseudo Class-2 validation.**
   Для validation-молекулы убираем её собственные спектры из spectral library, но оставляем структуру среди кандидатов. Это имитирует ситуацию «структура известна в базе, но эталонного спектра нет».

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

---

## Минимальный план запуска

### Шаг 1. Сделать validation split

Выбери 15–20% уникальных молекул как validation. Никогда не дели по строкам parquet.

```python
from src.validation import group_holdout_split

train_idx, val_idx = group_holdout_split(
    train_df,
    group_col='molecule_id',  # заменить на стабильный molecule key, если есть InChIKey
    val_fraction=0.20,
    random_state=42,
)
```

### Шаг 2. Построить pseudo Class-2 library

Для validation molecule нельзя использовать её собственные reference spectra.

```python
from src.validation import remove_validation_targets_from_library

library_df = remove_validation_targets_from_library(
    train_df.iloc[train_idx],
    val_df=train_df.iloc[val_idx],
    key_col='molecule_id',
)
```

Если split уже сделан корректно по molecule_id, эта функция почти ничего не меняет. Она оставлена как guard rail.

### Шаг 3. Получить несколько ranking lists

Не нужно полностью запускать pipeline 4 раза. Тяжёлые candidate features лучше посчитать один раз, а затем получить несколько score-векторов.

Пример views:

```python
views = {
    'entropy_002': scores_entropy_floor_002,
    'entropy_010': scores_entropy_floor_010,
    'entropy_020': scores_entropy_floor_020,
    'analog': scores_analog,
    'metfrag': scores_metfrag,
}
```

### Шаг 4. Объединить результаты

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

RRF не складывает raw scores. Для каждого view кандидат получает очки по позиции:

`weight / (k + rank)`

Это делает ensemble устойчивее, если у entropy score диапазон 0–1, а у другого метода шкала совсем другая.

### Шаг 5. Проверить метрики

```python
from src.validation import mrr_at_k, recall_at_k

print('Recall@25:', recall_at_k(predictions, truth, 25))
print('Recall@50:', recall_at_k(predictions, truth, 50))
print('Recall@100:', recall_at_k(predictions, truth, 100))
print('MRR@25:', mrr_at_k(predictions, truth, 25))
```

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

1. validation
2. candidate recall
3. multi-view preprocessing
4. rank fusion
5. ranker upgrade
6. formula gate
7. external candidate expansion
8. graph-edit generation

---

## Definition of Done для V17

V17 считается готовой, если:

- validation split не содержит одну и ту же молекулу в train и validation;
- есть pseudo Class-2 evaluation;
- считаются Recall@25/50/100 и MRR@25;
- есть минимум 3 score views;
- есть RRF ensemble;
- все параметры и результаты записываются в таблицу экспериментов;
- public Kaggle submission выполняется только после локального улучшения.
