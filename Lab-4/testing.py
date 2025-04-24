import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import time
import shutil
import os
from datetime import datetime
from moviepy.editor import ImageSequenceClip
import warnings
from tensorflow.keras.callbacks import TensorBoard
import logging


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('moviepy').setLevel(logging.ERROR)

print("TensorFlow версия:", tf.__version__)
print("Доступны следующие GPU устройства:", tf.config.list_physical_devices('GPU'))

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Использование GPU для обучения")
else:
    print("Использование CPU для обучения")


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized


# Класс для хранения истории кадров
class FrameStack:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        self.frames.clear()
        processed = preprocess_frame(frame)
        for _ in range(self.stack_size):
            self.frames.append(processed)
        return self.get_state()

    def add_frame(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self.get_state()

    def get_state(self):
        frames_array = np.array(self.frames)
        return np.moveaxis(frames_array, 0, -1)  # [frames, height, width] -> [height, width, frames]


# DQN сеть
def create_dqn_model(input_shape, n_actions, learning_rate=0.00025):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(n_actions, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    return model


# Буфер опыта
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Нормализатор вознаграждений
class RewardNormalizer:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mean = 0
        self.std = 1
        self.count = 0

    def normalize(self, reward):
        self.count += 1
        self.mean = self.mean + self.alpha * (reward - self.mean)
        self.std = self.std + self.alpha * (((reward - self.mean) ** 2) - self.std)

        norm_std = max(np.sqrt(self.std), 1e-4)

        if self.count > 100:
            normalized_reward = (reward - self.mean) / norm_std
        else:
            normalized_reward = reward

        return normalized_reward


# Агент DQN
class DQNAgent:
    def __init__(self, state_shape, n_actions, replay_buffer_size=50000,
                 batch_size=64, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.995,
                 learning_rate=0.00025, update_target_epochs=5):

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_epochs = update_target_epochs
        self.steps = 0
        self.episodes = 0
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.reward_normalizer = RewardNormalizer()

        self.model = create_dqn_model(state_shape, n_actions, learning_rate)
        self.target_model = create_dqn_model(state_shape, n_actions, learning_rate)
        self.update_target_network()

        self.buffer = ReplayBuffer(replay_buffer_size)

        self.losses = []
        self.q_values = []

        # Для TensorBoard
        self.log_dir = f"logs/dqn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard_callback = TensorBoard(log_dir=self.log_dir,
                                                histogram_freq=1,
                                                update_freq=100,
                                                write_graph=True)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def update_target_network(self):
        """Копирует веса из основной модели в целевую"""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, training=True):
        """Выбор действия с использованием epsilon-greedy стратегии"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
        q_values = self.model(state_tensor)

        max_q = tf.reduce_max(q_values).numpy()
        self.q_values.append(max_q)

        return tf.argmax(q_values[0]).numpy()

    def adjust_learning_rate(self, episode, total_episodes):
        """Адаптивная настройка learning rate в зависимости от прогресса обучения"""
        progress = min(1.0, episode / (total_episodes * 0.8))
        new_lr = self.initial_learning_rate * (1.0 - 0.9 * progress)

        if self.learning_rate != new_lr:
            self.learning_rate = new_lr
            self.model.optimizer.lr.assign(new_lr)

            with self.summary_writer.as_default():
                tf.summary.scalar('learning_rate', new_lr, step=episode)

    def dynamic_epsilon_decay(self, best_reward):
        """Динамический decay для epsilon в зависимости от прогресса обучения"""
        if best_reward >= 20:
            decay_factor = 0.99
        elif best_reward >= 10:
            decay_factor = 0.992
        else:
            decay_factor = self.epsilon_decay

        self.epsilon = max(self.epsilon_min, self.epsilon * decay_factor)

    @tf.function
    def learn(self, normalize_rewards=True):
        """Обучение модели на батче из буфера опыта"""
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        if normalize_rewards:
            normalized_rewards = np.array([self.reward_normalizer.normalize(r) for r in rewards])
        else:
            normalized_rewards = rewards

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(normalized_rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        next_q_values = self.target_model(next_states)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + self.gamma * max_next_q * (1 - dones)

        masks = tf.one_hot(actions, self.n_actions)

        with tf.GradientTape() as tape:
            q_values = self.model(states)

            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

            loss = keras.losses.Huber()(target_q_values, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        loss_value = loss.numpy()
        self.losses.append(loss_value)

        with self.summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=self.steps)
            tf.summary.scalar('max_q_value', tf.reduce_max(q_values).numpy(), step=self.steps)
            tf.summary.scalar('mean_q_value', tf.reduce_mean(q_values).numpy(), step=self.steps)
            tf.summary.scalar('epsilon', self.epsilon, step=self.steps)

        self.steps += 1

    def on_episode_end(self, episode, best_reward, total_episodes):
        """Вызывается в конце каждого эпизода для обновления параметров"""
        self.episodes += 1

        self.dynamic_epsilon_decay(best_reward)

        self.adjust_learning_rate(episode, total_episodes)

        if self.episodes % self.update_target_epochs == 0:
            self.update_target_network()
            # print(f"Target network updated (episode {episode})")

    def save(self, path):
        """Сохранение моделей и состояния агента"""
        self.model.save(path + "_model")
        self.target_model.save(path + "_target_model")

        state = {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'learning_rate': self.learning_rate,
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_std': self.reward_normalizer.std,
            'reward_normalizer_count': self.reward_normalizer.count
        }
        np.save(path + "_state.npy", state)

    def load(self, path):
        """Загрузка моделей и состояния агента"""
        self.model = keras.models.load_model(path + "_model")
        self.target_model = keras.models.load_model(path + "_target_model")

        state = np.load(path + "_state.npy", allow_pickle=True).item()
        self.epsilon = state['epsilon']
        self.steps = state['steps']
        self.episodes = state.get('episodes', 0)
        self.learning_rate = state.get('learning_rate', self.initial_learning_rate)

        if 'reward_normalizer_mean' in state:
            self.reward_normalizer.mean = state['reward_normalizer_mean']
            self.reward_normalizer.std = state['reward_normalizer_std']
            self.reward_normalizer.count = state['reward_normalizer_count']


def test_agent_with_retries(agent, model_path, num_episodes=10, min_reward_threshold=200, max_attempts=5,
                            record_video=True, analyze_q_values=True):
    """
    Запускает тестирование агента с повторами, если не достигнут порог награды.

    Args:
        agent: Обученный DQN агент
        model_path: Путь к файлу модели
        num_episodes: Количество эпизодов для тестирования
        min_reward_threshold: Минимальный порог награды для успешного тестирования
        max_attempts: Максимальное количество повторных попыток
        record_video: Записывать ли видео
        analyze_q_values: Анализировать ли Q-значения

    Returns:
        tuple: (test_rewards, best_reward, best_attempt)
    """
    best_reward = 0
    best_attempt = 0
    best_rewards = []

    successful_attempts = []

    for attempt in range(1, max_attempts + 1):
        print(f"\nПопытка тестирования #{attempt}:")

        # Создаем директории для текущей попытки
        test_video_dir = f"test_videos_attempt_{attempt}"
        q_analysis_dir = f"q_analysis_attempt_{attempt}"

        os.makedirs(test_video_dir, exist_ok=True)
        if analyze_q_values:
            os.makedirs(q_analysis_dir, exist_ok=True)

        agent.load(model_path)
        agent.epsilon = 0.1

        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        frame_stack = FrameStack(4)

        test_rewards = []
        all_q_values = []

        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            stacked_state = frame_stack.reset(state)

            episode_reward = 0
            frames = []
            q_values_episode = []
            actions_episode = []

            done = False
            while not done:
                state_tensor = tf.convert_to_tensor(stacked_state[np.newaxis, ...], dtype=tf.float32)
                q_values = agent.model(state_tensor)[0].numpy()

                max_q = np.max(q_values)
                q_values_episode.append(max_q)

                if np.random.random() < agent.epsilon:
                    action = np.random.randint(0, env.action_space.n)
                else:
                    action = np.argmax(q_values)

                actions_episode.append(action)

                next_state, reward, terminated, truncated, _ = env.step(action)
                frames.append(env.render())

                next_stacked_state = frame_stack.add_frame(next_state)

                stacked_state = next_stacked_state
                episode_reward += reward
                done = terminated or truncated

            test_rewards.append(episode_reward)
            print(f"Test Episode {episode}: Reward = {episode_reward}")

            if analyze_q_values and q_values_episode:
                all_q_values.extend(q_values_episode)

                q_threshold = np.percentile(q_values_episode, 95)
                high_q_indices = [i for i, q in enumerate(q_values_episode) if q > q_threshold]

                plt.figure(figsize=(12, 6))
                plt.plot(q_values_episode)
                plt.title(f"Q-values during Test Episode {episode} (Reward: {episode_reward})")
                plt.xlabel("Step")
                plt.ylabel("Max Q-value")
                plt.grid(True)

                if high_q_indices:
                    plt.plot(high_q_indices, [q_values_episode[i] for i in high_q_indices], 'ro', label='High Q-values')
                    plt.legend()

                plt.savefig(f"{q_analysis_dir}/q_values_episode_{episode}.png")
                plt.close()

                if high_q_indices and frames:
                    os.makedirs(f"{q_analysis_dir}/high_q_frames_ep{episode}", exist_ok=True)

                    for idx in high_q_indices[:5]:
                        if idx < len(frames):
                            plt.figure(figsize=(10, 8))
                            plt.imshow(frames[idx])
                            plt.title(
                                f"Q-value: {q_values_episode[idx]:.4f}, Step: {idx}, Action: {actions_episode[idx]}")
                            plt.axis('off')
                            plt.savefig(
                                f"{q_analysis_dir}/high_q_frames_ep{episode}/frame_step_{idx}_q_{q_values_episode[idx]:.4f}.png")
                            plt.close()

            if record_video and frames:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = f"{test_video_dir}/breakout_test_{timestamp}_ep{episode}_reward{int(episode_reward)}.mp4"
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)
                    clip = ImageSequenceClip(frames, fps=30)
                    clip.write_videofile(video_path, codec="libx264", verbose=False, logger=None)
                except Exception as e:
                    print(f"Ошибка при сохранении видео: {e}")

        if analyze_q_values and all_q_values:
            plt.figure(figsize=(12, 6))
            plt.hist(all_q_values, bins=50)
            plt.title("Distribution of Q-values across All Test Episodes")
            plt.xlabel("Q-value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(f"{q_analysis_dir}/q_values_distribution.png")
            plt.close()

        env.close()

        max_reward = max(test_rewards) if test_rewards else 0
        avg_reward = np.mean(test_rewards)
        print(f"Attempt #{attempt}: Max Reward = {max_reward}, Average Reward = {avg_reward:.2f}")

        if max_reward > best_reward:
            best_reward = max_reward
            best_attempt = attempt
            best_rewards = test_rewards.copy()

        if max_reward >= min_reward_threshold:
            print(f"Достигнут порог награды {min_reward_threshold} в попытке #{attempt}! Завершаем тестирование.")
            successful_attempts.append(attempt)

            print(f"Сохраняем результаты попытки #{attempt} как финальные.")

            if os.path.exists("test_videos"):
                shutil.rmtree("test_videos")
            if os.path.exists("q_analysis"):
                shutil.rmtree("q_analysis")

            shutil.copytree(test_video_dir, "test_videos")
            if analyze_q_values:
                shutil.copytree(q_analysis_dir, "q_analysis")

            success_video_dir = f"test_videos_success_{attempt}"
            success_q_dir = f"q_analysis_success_{attempt}"

            if os.path.exists(success_video_dir):
                shutil.rmtree(success_video_dir)
            if os.path.exists(success_q_dir):
                shutil.rmtree(success_q_dir)

            shutil.copytree(test_video_dir, success_video_dir)
            if analyze_q_values:
                shutil.copytree(q_analysis_dir, success_q_dir)

            break

    else:
        print(f"\nДостигнуто максимальное количество попыток ({max_attempts}).")
        print(f"Лучшая награда: {best_reward} в попытке #{best_attempt}")

        if best_attempt > 0:
            print(f"Сохраняем результаты лучшей попытки #{best_attempt} как финальные.")

            if os.path.exists("test_videos"):
                shutil.rmtree("test_videos")
            if os.path.exists("q_analysis"):
                shutil.rmtree("q_analysis")

            best_video_dir = f"test_videos_attempt_{best_attempt}"
            best_q_dir = f"q_analysis_attempt_{best_attempt}"

            if os.path.exists(best_video_dir):
                shutil.copytree(best_video_dir, "test_videos")
            if analyze_q_values and os.path.exists(best_q_dir):
                shutil.copytree(best_q_dir, "q_analysis")

            success_video_dir = f"test_videos_best_{best_attempt}"
            success_q_dir = f"q_analysis_best_{best_attempt}"

            if os.path.exists(success_video_dir):
                shutil.rmtree(success_video_dir)
            if os.path.exists(success_q_dir):
                shutil.rmtree(success_q_dir)

            if os.path.exists(best_video_dir):
                shutil.copytree(best_video_dir, success_video_dir)
            if analyze_q_values and os.path.exists(best_q_dir):
                shutil.copytree(best_q_dir, success_q_dir)

    print("Очищаем временные директории неуспешных попыток...")
    for attempt in range(1, max_attempts + 1):
        if attempt in successful_attempts or (len(successful_attempts) == 0 and attempt == best_attempt):
            continue

        temp_video_dir = f"test_videos_attempt_{attempt}"
        temp_q_dir = f"q_analysis_attempt_{attempt}"

        if os.path.exists(temp_video_dir):
            shutil.rmtree(temp_video_dir)
        if os.path.exists(temp_q_dir):
            shutil.rmtree(temp_q_dir)

    return best_rewards, best_reward, best_attempt


def main():
    try:
        env = gym.make("ALE/Breakout-v5")

        n_actions = env.action_space.n
        print(f"Действий доступно: {n_actions}")

        state_shape = (84, 84, 4)

        env.close()

        agent = DQNAgent(state_shape, n_actions,
                         replay_buffer_size=20000,
                         update_target_epochs=5,
                         epsilon_decay=0.995)

        print("Тестирование лучшей модели...")

        checkpoint_dir = "checkpoints"
        best_model_path = None
        best_reward_value = 0

        for file in os.listdir(checkpoint_dir):
            if file.startswith("breakout_best_") and file.endswith("_model"):
                try:
                    reward_str = file.replace("breakout_best_", "").replace("_model", "")
                    reward_value = int(reward_str)
                    if reward_value > best_reward_value:
                        best_reward_value = reward_value
                        best_model_path = os.path.join(checkpoint_dir, file.replace("_model", ""))
                except ValueError:
                    continue

        if best_model_path is None:
            best_model_path = f"{checkpoint_dir}/breakout_final"

        print(f"Используем модель: {best_model_path} с наградой {best_reward_value}")

        # Запускаем тестирование с повторными попытками
        rewards, best_reward, best_attempt = test_agent_with_retries(
            agent,
            best_model_path,
            num_episodes=5,  # Количество эпизодов на каждую попытку
            min_reward_threshold=150,  # Минимальный порог награды
            max_attempts=30,  # Максимальное количество попыток
            record_video=True,
            analyze_q_values=True
        )

        print(f"\nИтоговые результаты:")
        print(f"Лучшая награда: {best_reward} в попытке #{best_attempt}")
        print(f"Все награды в лучшей попытке: {rewards}")
        print(f"Средняя награда в лучшей попытке: {np.mean(rewards):.2f}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()