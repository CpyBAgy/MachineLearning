import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import warnings
import platform
from datetime import datetime
from IPython.display import Audio, display, HTML

warnings.filterwarnings("ignore")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 Используем GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print(f"🚀 Используем Apple Silicon GPU (MPS)")
else:
    print(f"⚠️ GPU не найден, используем CPU")

torch.set_num_threads(4)

RESULTS_DIR = "generated_music"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"🖥️  Система: {platform.system()} {platform.version()}")
print(f"🐍 Python: {platform.python_version()}")
print(f"🔥 PyTorch: {torch.__version__}")

def save_audio(audio_data, filename, model_name, prompt, sample_rate=32000):
    """Сохраняет аудио, показывает спектрограмму и отображает плеер"""
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()

    if len(audio_data.shape) > 1 and audio_data.shape[0] <= 2:
        audio_data = audio_data.T

    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        audio_data = np.clip(audio_data, -1.0, 1.0)

    sf.write(filename, audio_data, sample_rate)
    print(f"✅ Сохранено: {filename}")

    display(Audio(audio_data, rate=sample_rate))

    try:
        import librosa
        import librosa.display

        plt.figure(figsize=(12, 6))

        plt.suptitle(f"Модель: {model_name}", fontsize=16, y=0.99)
        plt.title(f"Промпт: '{prompt}'", fontsize=10)

        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sample_rate)
        plt.colorbar(format='%+2.0f dB')

        plot_filename = os.path.join(PLOTS_DIR, os.path.basename(filename).replace('.wav', '.png'))
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
    except ImportError:
        print("⚠️ librosa не установлен, пропускаем создание спектрограммы")

    return True

model_results = {}

# ===================== ФУНКЦИИ ГЕНЕРАЦИИ МУЗЫКИ =====================

def generate_with_musicgen(prompt="pop music with catchy chorus and electronic beats", duration=8):
    """Генерация музыки используя Meta's MusicGen Small модель"""
    model_name = "MusicGen Small"
    print(f"\n🎵 --- Генерация с помощью {model_name} ---")

    try:
        from audiocraft.models import MusicGen

        print("⚙️ Загружаем модель...")
        model = MusicGen.get_pretrained("facebook/musicgen-small")
        model.set_generation_params(duration=duration)

        descriptions = [prompt]

        print(f"🎼 Генерируем музыку по промпту: '{prompt}'...")
        wav = model.generate(descriptions, progress=True)

        for idx, (audio, desc) in enumerate(zip(wav, descriptions)):
            filename = f"{RESULTS_DIR}/musicgen_small_{idx+1}.wav"
            save_audio(audio[0], filename, model_name, desc, sample_rate=32000)

        return True
    except Exception as e:
        print(f"❌ Ошибка генерации с {model_name}: {e}")
        return False

def generate_with_audioldm(prompt="pop music with catchy chorus and electronic beats", duration=10):
    """Генерация музыки используя AudioLDM модель"""
    model_name = "AudioLDM"
    print(f"\n🎵 --- Генерация с помощью {model_name} ---")

    try:
        from diffusers import AudioLDMPipeline

        print("⚙️ Загружаем модель...")
        pipe = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-s-full-v2",
            torch_dtype=torch.float32 if device == "mps" else torch.float16
        )
        pipe = pipe.to(device)

        num_inference_steps = 10 if device == "cpu" else 50

        print(f"🎼 Генерируем музыку по промпту: '{prompt}'...")
        audio = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=duration
        ).audios[0]

        filename = f"{RESULTS_DIR}/audioldm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        save_audio(audio, filename, model_name, prompt, sample_rate=16000)

        return True
    except Exception as e:
        print(f"❌ Ошибка генерации с {model_name}: {e}")
        return False

def generate_with_musicldm(prompt="pop music with catchy melody and modern production", duration=5):
    """Генерация музыки используя MusicLDM модель"""
    model_name = "MusicLDM"
    print(f"\n🎵 --- Генерация с помощью {model_name} ---")

    try:
        from diffusers import DiffusionPipeline

        print("⚙️ Загружаем модель...")
        pipe = DiffusionPipeline.from_pretrained(
            "ucsd-reach/musicldm",
            torch_dtype=torch.float32
        )
        pipe = pipe.to(device)

        num_inference_steps = 10 if device == "cpu" else 50

        print(f"🎼 Генерируем музыку по промпту: '{prompt}'...")
        audio = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=duration
        ).audios[0]

        filename = f"{RESULTS_DIR}/musicldm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        save_audio(audio, filename, model_name, prompt, sample_rate=16000)

        return True
    except Exception as e:
        print(f"❌ Ошибка генерации с {model_name}: {e}")
        return False

def generate_with_audiogen(prompt="pop music with catchy chorus and electronic beats", duration=5):
    """Генерация музыки используя AudioGen модель"""
    model_name = "AudioGen"
    print(f"\n🎵 --- Генерация с помощью {model_name} ---")

    try:
        from audiocraft.models import AudioGen

        print("⚙️ Загружаем модель...")
        model = AudioGen.get_pretrained("facebook/audiogen-medium")
        model.set_generation_params(duration=duration)

        descriptions = [prompt]

        print(f"🎼 Генерируем музыку по промпту: '{prompt}'...")
        wav = model.generate(descriptions, progress=True)

        for idx, (audio, desc) in enumerate(zip(wav, descriptions)):
            filename = f"{RESULTS_DIR}/audiogen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            save_audio(audio[0], filename, model_name, desc, sample_rate=16000)

        return True
    except Exception as e:
        print(f"❌ Ошибка генерации с {model_name}: {e}")
        return False

def generate_with_musicgen_melody(prompt="pop music with catchy chorus and electronic beats", duration=8):
    """Генерация музыки используя MusicGen Melody модель"""
    model_name = "MusicGen Melody"
    print(f"\n🎵 --- Генерация с помощью {model_name} ---")

    try:
        from audiocraft.models import MusicGen

        print("⚙️ Загружаем модель...")
        model = MusicGen.get_pretrained("facebook/musicgen-melody")
        model.set_generation_params(duration=duration)

        descriptions = [prompt]

        print(f"🎼 Генерируем музыку по промпту: '{prompt}'...")
        wav = model.generate(descriptions, progress=True)

        for idx, (audio, desc) in enumerate(zip(wav, descriptions)):
            filename = f"{RESULTS_DIR}/musicgen_melody_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            save_audio(audio[0], filename, model_name, desc, sample_rate=32000)

        return True
    except Exception as e:
        print(f"❌ Ошибка генерации с {model_name}: {e}")
        return False

def generate_with_audioldm2(prompt="pop music with catchy chorus and electronic beats", duration=5):
    """Генерация музыки используя AudioLDM2 модель"""
    model_name = "AudioLDM2"
    print(f"\n🎵 --- Генерация с помощью {model_name} ---")

    try:
        from diffusers import AudioLDM2Pipeline

        print("⚙️ Загружаем модель...")
        pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=torch.float32
        )
        pipe = pipe.to(device)

        num_inference_steps = 10 if device == "cpu" else 50

        print(f"🎼 Генерируем музыку по промпту: '{prompt}'...")
        audio = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=duration
        ).audios[0]

        filename = f"{RESULTS_DIR}/audioldm2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        save_audio(audio, filename, model_name, prompt, sample_rate=16000)

        return True
    except Exception as e:
        print(f"❌ Ошибка генерации с {model_name}: {e}")
        return False



pop_prompt = "pop music with catchy chorus, electronic beats and uplifting melody"

print("🚀 Запускаем процесс генерации музыки...\n")

duration = 5 if device == "cpu" else 8

model_results["MusicGen Small"] = generate_with_musicgen(pop_prompt, duration=duration)

model_results["AudioLDM"] = generate_with_audioldm(pop_prompt, duration=duration)

model_results["MusicLDM"] = generate_with_musicldm(pop_prompt, duration=duration)

model_results["AudioGen"] = generate_with_audiogen(pop_prompt, duration=duration)

model_results["MusicGen Melody"] = generate_with_musicgen_melody(pop_prompt, duration=duration)

model_results["AudioLDM2"] = generate_with_audioldm2(pop_prompt, duration=duration)


print("\n\n📊 --- Итоги генерации ---")
successful_models = 0
results_table = "<table style='width:100%; border-collapse:collapse; margin:20px 0;'>"
results_table += "<tr style='background:#f2f2f2;'><th style='padding:10px; border:1px solid #ddd;'>Модель</th><th style='padding:10px; border:1px solid #ddd;'>Статус</th></tr>"

for model_name, success in model_results.items():
    status = "✅ УСПЕШНО" if success else "❌ ОШИБКА"
    status_color = "green" if success else "red"
    results_table += f"<tr><td style='padding:10px; border:1px solid #ddd;'>{model_name}</td>"
    results_table += f"<td style='padding:10px; border:1px solid #ddd; color:{status_color};'>{status}</td></tr>"

    if success:
        successful_models += 1

results_table += "</table>"

print(f"✅ Успешно сгенерировано: {successful_models}/{len(model_results)}")
display(HTML(results_table))

if successful_models < 3:
    print("\n⚠️ Сгенерировано менее 3 моделей. Возможные решения:")
    print("  1. Проверьте подключение к интернету для загрузки моделей")
    print("  2. Установите необходимые зависимости вручную (librosa, audiocraft, diffusers)")
    print("  3. Попробуйте запустить на GPU, если доступно")
    print("  4. Уменьшите длительность генерации (parameter duration)")
elif successful_models < len(model_results):
    print("\n⚠️ Некоторые модели не сработали. Попробуйте:")
    print("  1. Запустить их отдельно")
    print("  2. Проверить наличие необходимой памяти")
    print("  3. Уменьшить длительность генерации для моделей с ошибками")
else:
    print("\n🎉 Все модели успешно сгенерировали музыку!")