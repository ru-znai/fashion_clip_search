import os
from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# Установка модели и процессора
MODEL_NAME = "Margo-fashionCLIP"
IMAGE_FOLDER = "images/"

print("Загрузка модели CLIP...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("Модель загружена успешно.")


# Функция для загрузки изображений
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            images.append((img, filename))
    print(f"Загружено {len(images)} изображений из папки '{folder_path}'")
    return images


# Функция для поиска
def find_similar_image(text, images):
    # Преобразование текста в эмбеддинг с помощью CLIP
    print("Генерация эмбеддинга текста...")
    inputs = processor(text=[text], return_tensors="pt")
    text_features = model.get_text_features(**inputs)

    # Преобразование изображений
    print("Генерация эм... изображений...")
    image_embeddings = []
    for img, filename in images:
        img_input = processor(images=img, return_tensors="pt")
        image_features = model.get_image_features(**img_input)
        image_embeddings.append((image_features, filename))

    similarities = [(cosine_similarity(text_features, img_emb[0], dim=1).item(), img_emb[1]) for img_emb in
                    image_embeddings]
    most_similar_image = max(similarities, key=lambda x: x[0])

    return most_similar_image


# Главная функция
if __name__ == "__main__":

    images = load_images(IMAGE_FOLDER)
    if not images:
        print("В папке 'images/' не найдено изображений. Поместите изображения в эту папку и повторите запуск.")
        exit()
    text_query = input("Введите описание одежды для поиска (например, 'красная рубашка с длинными рукавами'): ")
    similar_image, similarity_score = find_similar_image(text_query, images)

    print(f"Наиболее похожее изображение: {similar_image} с оценкой сходства: {similarity_score:.4f}")

    img = Image.open(os.path.join(IMAGE_FOLDER, similar_image))
    img.show()
