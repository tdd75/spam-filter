import pandas as pd
import nltk
import math

nltk.download('stopwords')
nltk.download('wordnet')
stop = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

# Đọc file
df = pd.read_csv('emails.csv')

print(len(df))
print(df['Label'].value_counts(normalize=True))

# Chia tập dữ liệu theo tỉ lệ 8:2
sample = df.sample(frac=1, random_state=1)
index_split = round(len(sample) * 0.8)
df_train = sample[:index_split].reset_index(drop=True)
df_test = sample[index_split:].reset_index(drop=True)

print(len(df_train))
print(df_train['Label'].value_counts(normalize=True))
print(len(df_test))
print(df_test['Label'].value_counts(normalize=True))


# Tiền xử lý
def preprocess(df_col):
    df_col = df_col.str.replace('\W', ' ')  # Loại bỏ kí tự không phải chữ hoặc số
    df_col = df_col.apply(lambda doc: ' '.join([word for word in doc.split() if word not in stop]))  # Loại bỏ stopword
    df_col = df_col.apply(lambda doc: lemmatizer.lemmatize(doc))   # Chuẩn hóa từ
    df_col = df_col.str.lower()  # Đưa về dạng chữ thường
    df_col = df_col.str.split()  # Tách từ
    return df_col


df_train['Document'] = preprocess(df_train['Document'])

# Tạo feature vector
vocabulary = []
for doc in df_train['Document']:
    for word in doc:
        vocabulary.append(word)

vocabulary = list(set(vocabulary))
len(vocabulary)

# Đếm tần suất của mỗi từ trong từng thư
word_counts = {word: [0] * len(df_train['Document']) for word in vocabulary}
for index, doc in enumerate(df_train['Document']):
    for word in doc:
        word_counts[word][index] += 1

df_word_counts = pd.DataFrame(word_counts)
df_train = pd.concat([df_train, df_word_counts], axis=1)

# Tách DataFrame thành spam và ham
df_spam = df_train[df_train['Label'] == 1]
df_ham = df_train[df_train['Label'] == 0]

n_word_spam = df_spam['Document'].apply(len).sum()
n_word_ham = df_ham['Document'].apply(len).sum()
n_vocabulary = len(vocabulary)
alpha = 1  # Laplace smoothing

# Tính prior
p_spam = len(df_spam) / len(df_train)
p_ham = 1 - p_spam

# Tính likelihood
p_word_given_spam = {word: 0 for word in vocabulary}
p_word_given_ham = {word: 0 for word in vocabulary}

for word in vocabulary:
    n_word_given_spam = df_spam[word].sum()
    p_word_given_spam[word] = (n_word_given_spam + alpha) / (n_word_spam + alpha * n_vocabulary)
    n_word_given_ham = df_ham[word].sum()
    p_word_given_ham[word] = (n_word_given_ham + alpha) / (n_word_ham + alpha * n_vocabulary)


# Dự đoán một thư mới
def predict(doc):
    p_spam_given_words = math.log(p_spam)
    p_ham_given_words = math.log(p_ham)

    for word in doc:
        if word in vocabulary:
            p_spam_given_words += math.log(p_word_given_spam[word])
            p_ham_given_words += math.log(p_word_given_ham[word])

    if p_spam_given_words > p_ham_given_words:
        return 1
    else:
        return 0


# Dự đoán trên tập test
df_test['Document'] = preprocess(df_test['Document'])
df_test['Predict'] = df_test['Document'].apply(predict)

# Kết quả
TP = 0
TN = 0
FP = 0
FN = 0

for row in df_test.iterrows():
    row = row[1]
    if row['Label'] == row['Predict']:
        if row['Label'] == 1:
            TP += 1
        else:
            TN += 1
    else:
        if row['Label'] == 1:
            FP += 1
        else:
            FN += 1

print('TP:', TP)
print('TN:', TN)
print('FP:', FP)
print('FN:', FN)
print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))
print('Recall:', TP / (TP + FN))
print('Precision:', TP / (TP + FP))
