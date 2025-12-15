from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

csv_path = './csv/reviews_labeled.csv'
df = pd.read_csv(csv_path)
X = df.review.to_numpy()
Y = df.label.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.2, shuffle=True, random_state=42) 

baseline_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', LogisticRegression()) 
])

baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

print("Baseline Results:")

cm = confusion_matrix(y_test, y_pred_baseline)
print("\n--- Confusion Matrix ---")
print(cm)


print(classification_report(y_test, y_pred_baseline))

test_csv = './data_test/reviews_labeled.csv'
df = pd.read_csv(test_csv)

X = df.review.to_numpy()
Y = df.label.to_numpy()

y_test_pred = baseline_model.predict(X)
print("Baseline Test Data Results:")
print(classification_report(Y, y_test_pred))

cm = confusion_matrix(Y, y_test_pred)
print("\n--- Confusion Matrix ---")
print(cm)


df_report = pd.DataFrame(classification_report(Y, y_test_pred, output_dict=True)).transpose()

print(df_report)
df_report.to_excel("baseline_report.xlsx")