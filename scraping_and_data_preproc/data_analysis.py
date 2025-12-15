import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns

csv_path = '../csv/reviews_labeled.csv'
df = pd.read_csv(csv_path)

reviews_text = ' '.join(df['review'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.title('Word Cloud for Restaurant Reviews')
plt.show()


df['Review Length'] = df['review'].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 6))
sns.histplot(df['Review Length'], bins=100, kde=False, color='salmon')
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length (Number of Words)')
plt.ylabel('Count')
plt.show()