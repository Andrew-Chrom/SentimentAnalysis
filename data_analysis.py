import pandas as pd
import matplotlib.pyplot as plt

csv_data = './csv/reviews_labeled.csv'

df = pd.read_csv(csv_data)
print(df.head())

review_length = [len(review) for review in df.review]
plt.hist(review_length, bins=100)
plt.ylabel('length')
plt.show()