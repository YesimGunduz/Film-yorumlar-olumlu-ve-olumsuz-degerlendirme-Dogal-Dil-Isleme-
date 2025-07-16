import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import emoji



#burada yorumun gerekli düzeltmeleri ve emoji kelimeleştirme yaptık
def preprocess_text(text):
    # Etiket temizleme
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = emoji.demojize(text, language='tr')  #emojileri kelimeleştiriyoruz
    text = re.sub(r'[^a-zA-Z0-9\sçşğüıöÇŞĞÜİÖ]', ' ', text)
    text = text.lower() 
    

    nltk.download('stopwords', quiet=True) #önemsiz kelimeleri çıkarttık
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('turkish'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# dataseti yükledik
data = pd.read_csv('/Users/kaancankurt/Desktop/Bilgi Mühendisliğine giriş proje/yorumlar_5000.csv',sep=';')

# Etiketleri düzenledik
data.columns = data.columns.str.capitalize()
data['Yorum'] = data['Yorum'].apply(preprocess_text)

# Özellik ve etiketleri ayırdık
reviews = data['Yorum'].values
labels = data['Etiket'].values

# Label Encoding işlemi 1 ve 0 olarak negative positive kodladık
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Eğitim/Test seti %80 eğitim %20 test için ayırdık
train_reviews, test_reviews, train_labels, test_labels = train_test_split(
    reviews, encoded_labels, test_size=0.2, random_state=42, shuffle=True
)

# Vektörleştirme
cv = CountVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 3))
cv_train_reviews = cv.fit_transform(train_reviews).toarray()
cv_test_reviews = cv.transform(test_reviews).toarray()

# Model mimarisi
model = Sequential([
    Dense(128, activation='relu', input_shape=(cv_train_reviews.shape[1],),
          kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu', input_shape=(cv_train_reviews.shape[1],),
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.4),
    Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
history = model.fit(
    cv_train_reviews, train_labels,
    epochs=20,
    batch_size=32,
    validation_data=(cv_test_reviews, test_labels),
    callbacks=[early_stop],
    verbose=1
)


# TEST AŞAMASI VE PERFORMANSI
loss, accuracy = model.evaluate(cv_test_reviews, test_labels, verbose=0)
print(f"\n✅ Model Test Doğruluğu: {accuracy*100:.2f}%")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.tight_layout()
plt.show()

# KULLANICIDAN YORUM ALMA AŞAMASI
input_review = input("\nYorumunuzu Girin: ")

# YORUMU İŞLEME AŞAMASI
input_review_clean = preprocess_text(input_review)
input_vector = cv.transform([input_review_clean]).toarray()

# TAHMİN SONUÇLARI
prediction = model.predict(input_vector, verbose=0)
positive_prob = prediction[0][0]
negative_prob = 1 - positive_prob

# Tahmini belirle
if positive_prob > 0.5:
    predicted_label = 'positive'
    probability = positive_prob * 100
else:
    predicted_label = 'negative'
    probability = negative_prob * 100

# SONUÇ KISMI
print("\n--- Tahmin Sonucu ---")
print(f"Tahmin: {predicted_label.capitalize()} (%{probability:.2f})")


