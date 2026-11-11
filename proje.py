import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ==========================================
# 1. VERİYİ YÜKLEME VE TANIMA (İstenen 1)
# ==========================================
df = pd.read_csv('heart.csv')

# num > 0 ise kalp hastalığı var (1), 0 ise yok (0)
df['target'] = (df['num'] > 0).astype(int)

print("--- Verinin İlk 5 Satırı ---")
print(df.head())

print("\n--- Hedef Sınıf Dağılımı ---")
print(df['target'].value_counts()) # 0: Sağlıklı, 1: Hasta

# Analizde kullanılmayacak indeks ve kaynak sütunlarını çıkarıyoruz
df_clean = df.drop(columns=['id', 'dataset', 'num'])

# ==========================================
# 2. VERİ TEMİZLEME VE DÖNÜŞTÜRME (İstenen 2)
# ==========================================
# Sayısal sütunlardaki eksik değerleri (NaN) ortanca (median) değerle dolduruyoruz
num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.drop('target')
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Kategorik (metin) sütunlardaki eksik değerleri en sık geçen değerle dolduruyoruz
cat_cols = df_clean.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Metin olan sütunları (Male/Female, angina vb.) sayısal vektörlere dönüştürüyoruz (One-Hot Encoding)
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# ==========================================
# 3. GÖRSELLEŞTİRME (İstenen 3 - 5 Grafik)
# ==========================================
plt.figure(figsize=(14, 9))

# Grafik 1: Hastalık Dağılımı
plt.subplot(2, 3, 1)
sns.countplot(x='target', data=df_clean)
plt.title('1. Hedef Sınıf Dağılımı (0: Sağlıklı, 1: Hasta)')

# Grafik 2: Cinsiyete Göre Hastalık
plt.subplot(2, 3, 2)
sns.countplot(x='sex', hue='target', data=df_clean)
plt.title('2. Cinsiyete Göre Hastalık Oranı')

# Grafik 3: Yaş Dağılımı
plt.subplot(2, 3, 3)
sns.histplot(df_clean['age'], kde=True)
plt.title('3. Yaş Dağılımı')

# Grafik 4: Göğüs Ağrısı Tipi vs Hastalık
plt.subplot(2, 3, 4)
sns.countplot(x='cp', hue='target', data=df_clean)
plt.title('4. Göğüs Ağrısı Tipine Göre Hastalık')

# Grafik 5: Yaş ve Maksimum Nabız İlişkisi
plt.subplot(2, 3, 5)
sns.scatterplot(x='age', y='thalch', hue='target', data=df_clean)
plt.title('5. Yaş vs Maksimum Nabız (thalch)')

plt.tight_layout()
plt.show()

# ==========================================
# 4. MODEL EĞİTİMİ (İstenen 4)
# ==========================================
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Temel Model: Lojistik Regresyon
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==========================================
# 5. METRİKLER VE SONUÇLAR (İstenen 5)
# ==========================================
y_pred = model.predict(X_test)

print("\n================ METRİKLER (Logistic Regression) ================")
print(f"Accuracy (Doğruluk)  : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision (Kesinlik) : {precision_score(y_test, y_pred):.2f}")
print(f"Recall (Duyarlılık)  : {recall_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix (Karmaşıklık Matrisi):")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# BONUS: MODEL KARŞILAŞTIRMASI
# ==========================================
print("\n================ BONUS: MODEL KARŞILAŞTIRMASI ================")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    print(f"{name:20s} -> Accuracy: {acc:.2f} | Recall: {rec:.2f}")