import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONUTF8"] = "1"

# --- 1. Завантаження SavedModel ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_attention_full") 

print(f"Завантаження моделі з {MODEL_PATH}...")

try:
    loaded_obj = tf.saved_model.load(MODEL_PATH)
    inference_func = loaded_obj.signatures["serving_default"]
    print("✅ Модель успішно завантажена!")
except Exception as e:
    print(f"Помилка завантаження: {e}")
    exit()

# --- 2. Підготовка даних ---
CSV_PATH = "./csv/reviews_labeled.csv" 
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "../csv/reviews_labeled.csv")

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print("Не знайдено файл reviews_labeled.csv")
    exit()

x = df['review'].astype(str).to_numpy()
y = df['label'].astype(int).to_numpy()

_, x_test, _, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

# --- 3. Спеціальна функція для передбачення ---
def custom_predict(texts, batch_size=32):
    results = []
    total = len(texts)
    print(f"🔄 Обробка {total} відгуків...")
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        
        # 1. Створюємо тензор (це буде shape=(32,))
        inp = tf.constant(batch)
        
        # 2. !!! ВАЖЛИВЕ ВИПРАВЛЕННЯ !!!
        # Додаємо вимір, щоб стало shape=(32, 1)
        inp = tf.expand_dims(inp, axis=-1)
        
        # Викликаємо функцію моделі
        raw_output = inference_func(inp)
        
        # Отримуємо результат
        out_tensor = list(raw_output.values())[0]
        results.extend(out_tensor.numpy())
        
    return np.array(results)

reviews = [['Це просто жах, а не ресторан'], ['Це просто жах'], ['Сервіс був настільки швидким, що я встиг постаріти']]

# predictions = custom_predict(reviews)

for review in reviews:
    print(f'Review: `{review[0]}`')
    prediction = custom_predict(review)
    print(f'Prediction: {prediction}')

# # --- 4. Виконання ---
# try:
#     y_pred_proba = custom_predict(x_test)
# except Exception as e:
#     print(f"\nПомилка під час передбачення: {e}")
#     exit()

# # Бінаризація
# y_pred = (y_pred_proba > 0.5).astype(int)

# # --- 5. Метрики та графіки ---
# cm = confusion_matrix(y_test, y_pred)
# print("\n--- Confusion Matrix ---")
# print(cm)

# class_names = ['Negative', 'Positive']
# plt.figure(figsize=(8, 6))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names, rotation=45)
# plt.yticks(tick_marks, class_names)

# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, format(cm[i, j], 'd'),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

# plt.tight_layout()
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# save_path = os.path.join(BASE_DIR, 'confusion_matrix_final.png')
# plt.savefig(save_path)
# print(f"✅ Графік збережено: {save_path}")

# print("\n--- Classification Report ---")
# print(classification_report(y_test, y_pred, target_names=class_names))


# # check: 