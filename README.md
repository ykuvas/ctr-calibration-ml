# ctr-calibration-ml
CTR prediction with linear models and probability calibration (isotonic); PR-AUC/LogLoss/Brier, sklearn pipelines and artifact saving.
# CTR prediction + probability calibration (isotonic)

Проект: предсказание вероятности клика по рекламному показу (CTR) на табличных данных и улучшение качества вероятностей с помощью калибровки.

## Goal
- Построить интерпретируемую модель, предсказывающую `P(click=1)` для рекламных показов.
- Оценить качество ранжирования и качество вероятностей.
- Выполнить калибровку вероятностей (isotonic) на отдельной калибровочной выборке.

## Data
- Табличные данные рекламных показов: **50 000 строк, 34 признака**
- Целевая переменная: `click` (0/1)
- Источник датасета (для воспроизведения):  
  `https://code.s3.yandex.net/datasets/ds_s16_ad_click_dataset.csv`

## Approach
- Baseline: `DummyClassifier`
- Модели: `LogisticRegression`, `SVC(kernel="linear")`
- Валидация: cross-validation, подбор гиперпараметров `GridSearchCV`
- Кодирование категориальных признаков:
  - OHE для признаков с низкой кардинальностью
  - Target Encoding для высококардинальных признаков (`category_encoders`)
- Калибровка вероятностей:
  - `CalibratedClassifierCV(method="isotonic")` на отдельной calibration split
  - сравнение до/после по Brier score и calibration curve

## Metrics
- Основная: **PR-AUC** (актуально при дисбалансе классов)
- Дополнительные: **LogLoss**, **Brier score**
- Диагностика калибровки: calibration curve (опционально ECE/MCE)

## Project structure
- `ctr_prediction_calibration.ipynb` — основной ноутбук
- `requirements.txt` — зависимости
- `data/` — папка для локального датасета (не хранится в репозитории)

## How to run
### Option A: run with local dataset
1. Скачайте датасет по ссылке выше и положите в `data/`:
   - `data/ds_s16_ad_click_dataset.csv`
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
