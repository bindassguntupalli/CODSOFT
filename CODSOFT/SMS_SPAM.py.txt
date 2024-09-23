import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import string

data = pd.read_csv('C:/Users/chara/Downloads/Spamdataset/spam.csv', encoding='latin-1')

print(data.head())
print(data.columns)

data.columns = ['label', 'message', 'extra1', 'extra2', 'extra3']
data.drop(columns=['extra1', 'extra2', 'extra3'], inplace=True)

# Check and rename columns if necessary
if 'label' not in data.columns or 'message' not in data.columns:
    data.columns = ['label', 'message', 'other_columns']

data.dropna(subset=['message', 'label'], inplace=True)


def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text


data['message'] = data['message'].apply(preprocess_text)

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB())

# Training the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


def classify_message(message):
    message = preprocess_text(message)
    prediction = model.predict([message])
    return prediction[0]


print(classify_message("Congratulations! You've won a free ticket!"))
print(classify_message("Are we still meeting for lunch?"))

# Expected output spam and ham