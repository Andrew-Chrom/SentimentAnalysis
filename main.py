"""
Main file, where I load reviews.csv(convert in reviews_labeled.csv) 
and training simple model
"""

import os

# there is problem with ukrainian encoding
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONUTF8"] = "1"


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# importing training data
csv_path = "./csv/reviews_labeled.csv"
df = pd.read_csv(csv_path)
print(df.head())

x = df['review'].astype(str).to_numpy()
y = df['label'].astype(int).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

tf.random.set_seed(42)

text_vectorizer = tf.keras.layers.TextVectorization(
    output_mode='int',
    output_sequence_length=200)

text_vectorizer.adapt(df.review)
embeddings = tf.keras.layers.Embedding(input_dim=len(text_vectorizer.get_vocabulary()),
                                       output_dim=128)
inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embeddings(x)
x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
attn_output = tf.keras.layers.Attention()([x, x])
x = tf.keras.layers.GlobalAveragePooling1D()(attn_output)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)


model = tf.keras.Model(inputs, outputs)

from tensorflow.keras.utils import plot_model

plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=200
)
# model.compile(loss=tf.keras.losses.binary_crossentropy,
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])

# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights_dict = dict(enumerate(class_weights))

# print(model.summary())

# EPOCHS = 12

# history = model.fit(
#     x_train, y_train,
#     epochs=EPOCHS,
#     validation_data=(x_test, y_test),
#     class_weight=class_weights_dict,
#     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
# )

# os.makedirs("./models", exist_ok=True)
# model.export("models/model_attention_full")
# df = pd.DataFrame(history.history)
# df.to_csv('./training_history.csv')

# history = history.history
# # print(len(history))
# epochs_ran = len(np.squeeze(history['loss']))
# range_of_epochs = range(1, epochs_ran + 1)


# figure1 = plt.figure(figsize=(12, 6))
# ax1 = figure1.add_subplot()
# ax1.plot(range_of_epochs, history['loss'])
# ax1.plot(range_of_epochs, history['val_loss'])
# ax1.legend(['Training Loss', 'Validation Loss'])
# figure1.savefig('loss.png')

# figure2 = plt.figure(figsize=(12, 6))
# ax2 = figure2.add_subplot()
# ax2.plot(range_of_epochs, history['accuracy'])
# ax2.plot(range_of_epochs, history['val_accuracy'])
# ax2.legend(['Training Accuracy', 'Validation Accuracy'])
# figure2.savefig('accuracy.png')

# # model.save("./models/model_attention_full.keras")
# # model.save_weights('model_attention.weights.h5')

# # model.save('model_attention_full.keras')


# test_csv_path = './data_test/reviews_labeled.csv'
# df_test = pd.read_csv(test_csv_path, encoding='utf-8-sig')

# x_test = df_test['review'].astype(str).to_numpy()
# y_test = df_test['label'].astype(int).to_numpy()



# results = model.evaluate(x_test, y_test, batch_size=32)
# print(model.predict(tf.constant(["Це просто жах"])))

# print(f"\nTest loss: {results[0]:.4f}")
# print(f"Test accuracy: {results[1]:.4f}")




# # csv_path = "./csv/reviews.csv"
# # df = pd.read_csv(csv_path)

# # print(df.head())

# # def label_from_rating(rating):
# #     if rating in [1, 2]:
# #         return 0   # негатив
# #     elif rating == 5:
# #         return 1   # позитив
# #     else:
# #         return None  # нейтральні або непотрібні


# # df['label'] = df['rating'].apply(label_from_rating)

# # df = df.dropna(subset=['label']).reset_index(drop=True)
# # df['label'] = df['label'].astype(int)

# # print(df.head())
# # print(df['label'].value_counts())


# # df.to_csv('./csv/reviews_labeled.csv', index=False, encoding='utf-8-sig')


