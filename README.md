# ML FastAPI

Учебный сервис предсказывает, уйдёт ли клиент в следующем месяце. Модель возвращает класс `churn` и вероятности обоих классов. Класс `1` означает отток, класс `0` — клиент остался.

## Датасет

Положите `churn_dataset.csv` в корень проекта. Первая строка CSV содержит заголовки.

| Столбец | Тип | Значение |
| --- | --- | --- |
| `monthly_fee` | `float` | Ежемесячная стоимость тарифа |
| `usage_hours` | `float` | Часы использования за последний месяц |
| `support_requests` | `int` | Количество обращений в поддержку |
| `account_age_months` | `int` | Возраст аккаунта в месяцах |
| `failed_payments` | `int` | Количество неудачных платежей |
| `region` | `str` | `europe`, `asia`, `america`, `africa` |
| `device_type` | `str` | `mobile`, `desktop`, `tablet` |
| `payment_method` | `str` | `card`, `paypal`, `crypto` |
| `autopay_enabled` | `int` | `1` — автосписание включено, `0` — выключено |
| `churn` | `int` | Целевой класс `0` или `1` |

Сервис делит данные на обучающую (80%) и тестовую (20%) выборки со стратификацией. Pipeline заполняет пропуски в признаках по обучающей выборке, масштабирует числовые признаки и кодирует категории.

## Локальный запуск

Из корня проекта с Python 3.14 и активированным виртуальным окружением выполните команды.

```bash
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Swagger UI доступен по адресу [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Docker

Выполните команды из корня проекта. Контейнер читает локальный CSV, а Docker-том хранит модель между запусками.

```bash
docker build -t ml-fastapi:local .
docker run --rm -p 127.0.0.1:8000:8000 \
  --mount "type=bind,source=$PWD/churn_dataset.csv,target=/app/churn_dataset.csv,readonly" \
  --mount type=volume,source=ml-fastapi-models,target=/app/models \
  ml-fastapi:local
```

`GET /health` показывает, доступна ли модель и читается ли непустой датасет.

## Обучение и прогноз

Запрос без тела обучает логистическую регрессию с настройками по умолчанию.

```bash
curl -X POST http://127.0.0.1:8000/model/train
```

Чтобы выбрать случайный лес и передать его параметры, отправьте JSON.

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"random_forest","hyperparameters":{"n_estimators":100,"max_depth":5}}'
```

Ответ содержит `accuracy` и `f1` по тестовой выборке. `GET /model/status` показывает параметры последней модели, `GET /model/metrics` — результаты последних пяти обучений. После перезапуска сервис восстанавливает модель и метрики последнего обучения из `models/churn_model.joblib`.

После обучения передайте признаки клиента в `POST /predict`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "monthly_fee": 19.99,
    "usage_hours": 21.48,
    "support_requests": 2,
    "account_age_months": 1,
    "failed_payments": 0,
    "region": "america",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
```

Пример ответа

```json
{"churn": 0, "probabilities": {"0": 0.82, "1": 0.18}}
```

Для пакетного прогноза отправьте непустой массив объектов. Список признаков и их типы доступны по `GET /model/schema`.

## Тесты

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```
