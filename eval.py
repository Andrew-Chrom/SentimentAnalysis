import tensorflow as tf
import pandas as pd
import numpy as np

csv_path = "./csv/reviews_labeled.csv"
df = pd.read_csv(csv_path)
# print(df.head())

text_vectorizer = tf.keras.layers.TextVectorization(
    output_mode='int',                # перетворюємо в цілочислові токени
    output_sequence_length=200)

text_vectorizer.adapt(df.review)

embeddings = tf.keras.layers.Embedding(input_dim=len(text_vectorizer.get_vocabulary()),
                                       output_dim=128)


inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embeddings(x)
x = tf.keras.layers.LSTM(64)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)


model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

model.load_weights("mode.weights.h5")


data = ["""Вимагати 10% чайових від суми чеку - це наглість вищого рівня. Якщо , як ви кажете, це правила закладу, тоді ці 10% мають відобразитися в чеку, але ні, ви просто їх вимагаєте без будь-яких підстав. Я не проти заплатити чайові, але їх розмір- це моє право, а не ваша нахабність з вимаганням.

Нижче відповідь управляючої, яка пропонує їй зателефонувати , але залишила недійсним номер))""", 
"Тримати папуг в кутку заради декору в сучасному світі це дурний тон.", 
'Повар жінка, яка готовить хачапури, ковиряється в зубах, якщо не вміють соблюдать гігіену, нехай готують в рукавичках. Це жах.',

"Обслуговування просто лааааввв, дуже чуйні, уважні та дуже сервісні ❤️",
"Атмосфера супер! Кухня - топ! офіціант Владислава - 10 з 10",
"Дякую, все чудово і смачно. Дякую, Влада, за обслуговування!"
]

# data = np.array(data, dtype=str).reshape(-1, 1)

# labels = np.array([0,0,0, 1, 1, 1], dtype=int)

data = tf.constant(data, tf.string)
labels = tf.constant([0,0,0, 1, 1, 1], dtype=tf.int32)

print(model.summary())