import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import glob

MAX_EPISODES = 4000

logs_dir = "logs"
plots_dir = "plots"

os.makedirs(plots_dir, exist_ok=True)


def load_tensorboard_data(log_dir):
    data = {}
    for event_file in glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True):
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                if v.tag not in data:
                    data[v.tag] = []
                try:
                    data[v.tag].append((e.step, float(tf.make_ndarray(v.tensor))))
                except:
                    pass

    for tag in data:
        data[tag] = sorted(data[tag], key=lambda x: x[0])

    return data


def plot_metrics_limited(data):
    plot_labels = {
        'episode_reward': ('Награда за эпизод', 'Эпизод', 'Награда'),
        'best_reward': ('Лучшая достигнутая награда', 'Эпизод', 'Лучшая награда'),
        'avg_reward_100ep': ('Средняя награда за 100 эпизодов', 'Эпизод', 'Средняя награда'),
        'epsilon': ('Значение эпсилон (исследование/эксплуатация)', 'Шаг', 'Эпсилон'),
        'loss': ('Функция потерь', 'Шаг', 'Потери'),
        'learning_rate': ('Скорость обучения', 'Эпизод', 'Learning Rate'),
        'max_q_value': ('Максимальные Q-значения', 'Шаг', 'Max Q-значение'),
        'mean_q_value': ('Средние Q-значения', 'Шаг', 'Mean Q-значение')
    }

    for tag, values in data.items():
        if len(values) == 0 or tag not in plot_labels:
            continue

        title, xlabel, ylabel = plot_labels[tag]

        if 'episode' in tag or 'reward' in tag or 'learning_rate' in tag:
            filtered_values = [(step, value) for step, value in values if step <= MAX_EPISODES]
        else:
            filtered_values = values

        if not filtered_values:
            continue

        steps, y_values = zip(*filtered_values)
        plt.figure(figsize=(12, 6))
        plt.plot(steps, y_values, 'b-', label='Значения')

        if len(y_values) > 10:
            window_size = min(100, len(y_values) // 10)
            moving_avg = np.convolve(y_values, np.ones(window_size) / window_size, mode='valid')
            plt.plot(steps[window_size - 1:window_size - 1 + len(moving_avg)], moving_avg,
                     'r--', linewidth=2, label='Скользящее среднее')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()

        if tag == 'episode_reward':
            plt.text(0.05, 0.95, 'Награда, полученная за каждый эпизод обучения',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        elif tag == 'best_reward':
            plt.text(0.05, 0.95, 'Наилучший результат, достигнутый к данному эпизоду',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        elif tag == 'avg_reward_100ep':
            plt.text(0.05, 0.95, 'Скользящее среднее за последние 100 эпизодов',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        elif tag == 'epsilon':
            plt.text(0.05, 0.95, 'Вероятность случайного действия (исследование)',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        elif tag == 'loss':
            plt.text(0.05, 0.95, 'Значение функции потерь при обучении сети',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        elif tag == 'learning_rate':
            plt.text(0.05, 0.95, 'Адаптивная скорость обучения модели',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        elif tag == 'max_q_value':
            plt.text(0.05, 0.95, 'Максимальные Q-значения указывают на уверенность модели',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
        elif tag == 'mean_q_value':
            plt.text(0.05, 0.95, 'Средние Q-значения для всех действий',
                     transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')

        plt.savefig(f"{plots_dir}/{tag.replace('/', '_')}.png", dpi=300, bbox_inches='tight')
        print(f"Сохранен график: {tag}")
        plt.close()


def main():
    print("Загрузка данных из TensorBoard...")
    tb_data = load_tensorboard_data(logs_dir)

    print("Построение графиков с ограничением до 2000 эпизодов...")
    plot_metrics_limited(tb_data)

    print(f"\nВсе графики сохранены в директории: {plots_dir}")


if __name__ == "__main__":
    main()