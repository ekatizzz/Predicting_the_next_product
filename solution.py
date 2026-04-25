"""Афанасьева Екатерина
Рекомендательная система: предсказание следующего товара в сессии."""

import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# ЗАГРУЗКА ДАННЫХ
def load_sessions(path):
    """ читаем файл построчно, потому что там JSON Lines """
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


# АНАЛИЗ ДАННЫХ
def data_analyze(sessions):
    """ считаем базовые характеристики датасета """
    n_sessions = len(sessions)
    lengths = [len(s) for s in sessions]

    # собираем все товары в один список
    all_items = [item for s in sessions for item in s]
    unique_items = set(all_items)

    # частоты товаров
    item_counts = Counter(all_items)

    print("АНАЛИЗ")
    print("Число сессий:", n_sessions)
    print("Уникальных товаров:", len(unique_items))
    print("Средняя длина:", sum(lengths) / len(lengths))
    print("Минимальная длина:", min(lengths))
    print("Максимальная длина:", max(lengths))

    print("\nТоп 10 популярных товаров:")
    for item, cnt in item_counts.most_common(10):
        print(item, cnt)

    # простая гистограмма длин сессий
    plt.figure()
    plt.hist(lengths, bins=20)
    plt.title("Распределение длин сессий")
    plt.xlabel("Длина")
    plt.ylabel("Кол-во")
    plt.show()

# TRAIN
def train_test_split(sessions):
    """ цель - последний товар в каждой сессии """
    train_sessions = [s[:-1] for s in sessions]
    test_targets = [s[-1] for s in sessions]
    return train_sessions, test_targets

# ГРАФ ПЕРЕХОДОВ
def build_model(train_sessions):
    """ считаем, как часто один товар переходит в другой """
    transitions = defaultdict(Counter)

    for session in train_sessions:
        for i in range(len(session) - 1):
            curr_item = session[i]
            next_item = session[i + 1]
            transitions[curr_item][next_item] += 1

    # переводим счетчики в вероятности
    probs = {}
    for item, counter in transitions.items():
        total = sum(counter.values())
        probs[item] = {
            nxt: cnt / total
            for nxt, cnt in counter.items()
        }

    return probs

# ПОПУЛЯРНЫЕ ТОВАРЫ
def get_popular(train_sessions, k=10):
    """ если товар неизвестен, используем самые частые товары """
    all_items = [item for s in train_sessions for item in s]
    return [item for item, _ in Counter(all_items).most_common(k)]

# РЕКОМЕНДАЦИИ
def recommend_next_items(session, model, popular_items, k=10):
    """ если такой товар уже встречался, берем вероятные переходы """
    last_item = session[-1]

    if last_item in model:
        ranked = sorted(
            model[last_item].items(),
            key=lambda x: x[1],
            reverse=True
        )
        recs = [item for item, _ in ranked if item != last_item]
    else:
        recs = []

    # если не хватило рекомендаций, добираем популярными товарами
    for item in popular_items:
        if item not in recs and item != last_item:
            recs.append(item)

    return recs[:k]

# Hit@K
def hit_at_k(recommendations, true_items, k=10):
    """ считаем, как часто правильный товар попал в top-k """
    assert len(recommendations) == len(true_items)

    hits = 0
    for recs, true_item in zip(recommendations, true_items):
        if true_item in recs[:k]:
            hits += 1

    return hits / len(true_items)

# ОЦЕНКА МОДЕЛИ
def evaluate_model(model, train_sessions, test_targets):
    """ здесь тоже используем популярные товары как запасной вариант """
    popular = get_popular(train_sessions, k=10)

    recommendations = [
        recommend_next_items(s, model, popular, k=10)
        for s in train_sessions
    ]

    return hit_at_k(recommendations, test_targets, k=10)

# BASELINE
def evaluate_baseline(train_sessions, test_targets):
    """ рекомендуем только популярные товары """
    popular = get_popular(train_sessions, k=10)
    recommendations = [popular for _ in train_sessions]
    return hit_at_k(recommendations, test_targets, k=10)

# MAIN
def main():
    """ основной код, который запускается при выполнении скрипта """
    path = "sessions.jsonl"

    sessions = load_sessions(path)

    data_analyze(sessions)

    train_sessions, test_targets = train_test_split(sessions)

    model = build_model(train_sessions)

    model_score = evaluate_model(model, train_sessions, test_targets)
    baseline_score = evaluate_baseline(train_sessions, test_targets)

    print("\nРЕЗУЛЬТАТЫ")
    print("Model Hit@10:", round(model_score, 4))
    print("Baseline Hit@10:", round(baseline_score, 4))

    # если модель хуже бейзлайна, это нужно отметить
    if model_score < baseline_score:
        print("\n⚠ Модель хуже бейзлайна, нужно улучшать!")

if __name__ == "__main__":
    main()
