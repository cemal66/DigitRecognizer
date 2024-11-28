import pandas as pd

# Veri dosyalarını yükleyelim
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Veri boyutlarını inceleyelim
print("Eğitim verisi boyutları:", train_data.shape)
print("Test verisi boyutları:", test_data.shape)

# İlk birkaç satıra bakalım
print("\nEğitim Verisi İlk Satırlar:")
print(train_data.head())

print("\nTest Verisi İlk Satırlar:")
print(test_data.head())

input("Devam etmek için Enter'a basın...")

import matplotlib.pyplot as plt
import seaborn as sns

# Etiketlerin dağılımını görselleştirelim
plt.figure(figsize=(8, 6))
sns.countplot(x=train_data['label'], hue=None, color='blue')  # palette yerine color kullanıldı
plt.title("Etiketlerin Dağılımı (Label)")
plt.xlabel("Sınıf")
plt.ylabel("Frekans")
plt.show()

# Görüntü verisinden bir örnek görselleştirelim
example = train_data.iloc[0, 1:].values.reshape(28, 28)  # İlk görüntü
plt.figure(figsize=(6, 6))
plt.imshow(example, cmap='gray')
plt.title(f"Örnek Görüntü - Label: {train_data.iloc[0, 0]}")
plt.axis('off')
plt.show()

input("Devam etmek için Enter'a basın...")

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Özellikler (X) ve etiketler (y)
X = train_data.iloc[:, 1:].values  # Görüntü pikselleri
y = train_data.iloc[:, 0].values  # Etiketler

# Veriyi normalize et (pikselleri [0, 1] aralığına getir)
X = X / 255.0

# Etiketleri one-hot encode et
y = to_categorical(y, num_classes=10)

# Eğitim ve doğrulama setlerine ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Eğitim verisi boyutu:", X_train.shape, y_train.shape)
print("Doğrulama verisi boyutu:", X_val.shape, y_val.shape)

input("Devam etmek için Enter'a basın...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Modeli oluştur
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Giriş katmanı
    Dropout(0.2),  # Overfitting'i önlemek için Dropout
    Dense(64, activation='relu'),  # Orta katman
    Dropout(0.2),
    Dense(10, activation='softmax')  # Çıkış katmanı
])

# Modeli derle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modelin özetini yazdır
model.summary()

input("Devam etmek için Enter'a basın...")

# Modeli eğit
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Eğitim döngüsü sayısı
    batch_size=32,  # Her iterasyonda kullanılacak veri boyutu
    verbose=1  # Eğitim ilerlemesi gösterimi
)

input("Devam etmek için Enter'a basın...")

# Eğitim süreci görselleştirme
plt.figure(figsize=(12, 5))

# Doğruluk (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title('Doğruluk (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title('Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.show()

input("Devam etmek için Enter'a basın...")

# Test verilerini normalize et
X_test = test_data.values / 255.0

# Tahmin yap
predictions = model.predict(X_test)
predicted_labels = predictions.argmax(axis=1)  # En yüksek olasılıklı sınıfı seç

# Sonuçları sample_submission formatına uygun hale getir
output = sample_submission.copy()
output['Label'] = predicted_labels

# Çıktıyı kaydet
output.to_csv('submission.csv', index=False)
print("Sonuçlar 'submission.csv' dosyasına kaydedildi!")

print(output.head())

# Ayrıca test verilerinin ilk birkaç tahminini görselleştirelim
for i in range(5):  # İlk 5 tahmini görselleştirelim
    plt.figure(figsize=(3, 3))
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Tahmin: {predicted_labels[i]}")
    plt.axis('off')
    plt.show()

