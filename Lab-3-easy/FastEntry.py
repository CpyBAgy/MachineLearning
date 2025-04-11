import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def load_model_and_mapping():
    with open('id_to_location.json', 'r') as f:
        id_to_location = json.load(f)
        id_to_location = {int(k): v for k, v in id_to_location.items()}

    device_for_loading = torch.device("cpu")
    print("Загрузка модели на CPU...")

    if torch.cuda.is_available():
        inference_device = torch.device("cuda")
        print("После загрузки будет использоваться GPU (CUDA)")
    elif torch.backends.mps.is_available():
        inference_device = torch.device("mps")
        print("После загрузки будет использоваться MPS (Apple GPU)")
    else:
        inference_device = torch.device("cpu")
        print("Будет использоваться CPU")

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    num_labels = len(id_to_location)
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels
    )

    model_state = torch.load('roberta_coco_locations_best.pt',
                             map_location=device_for_loading)
    model.load_state_dict(model_state)

    model = model.to(inference_device)
    model.eval()

    return model, tokenizer, id_to_location, inference_device


def predict_location(text, model, tokenizer, id_to_location, device):
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return id_to_location[pred]


def main():
    print("Загрузка модели...")
    try:
        model, tokenizer, id_to_location, device = load_model_and_mapping()
        print("Модель успешно загружена!")
        print("\nВведите текст для определения локации (или 'выход' для завершения):")

        while True:
            text = input("> ")
            if text.lower() in ['quit', 'q']:
                break

            if not text:
                print("Пожалуйста, введите текст.")
                continue

            location = predict_location(text, model, tokenizer, id_to_location, device)
            print(f"Предсказанная локация: {location}")
    except Exception as e:
        print(f"Ошибка при загрузке или работе с моделью: {e}")
        print("\nПопробуйте следующие решения:")
        print("1. Принудительно использовать только CPU для загрузки и инференса")
        print("2. Проверить, что файл модели не поврежден")
        print("3. Пересохранить модель, выполнив следующий код:")
        print("""
# Код для пересохранения модели:
import torch
model_state = torch.load('roberta_coco_locations_best.pt', map_location=torch.device('cpu'))
torch.save(model_state, 'roberta_coco_locations_best_fixed.pt')
""")


if __name__ == "__main__":
    main()