import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pyvista as pv
from sklearn.metrics import silhouette_score

# Dosya yolu
file_path = 'wp_wc_order_product_lookup.csv'

# Veriyi yükleme
orders_df = pd.read_csv(file_path)

# 'product_id' ile 'category' eşlemesi
category_mapping = {
    20: 'giyim', 22: 'giyim', 23: 'giyim', 24: 'giyim', 179: 'aksesuar',
    180: 'aksesuar', 181: 'aksesuar', 182: 'giyim', 183: 'giyim', 184: 'giyim',
    185: 'giyim', 186: 'giyim', 210: 'aksesuar', 487: 'giyim', 499: 'elektronik',
    502: 'elektronik', 504: 'elektronik', 506: 'elektronik', 508: 'mutfak',
    510: 'mutfak', 512: 'mutfak', 514: 'mutfak', 516: 'beyazEsya',
    518: 'beyazEsya', 520: 'kozmetik', 522: 'kozmetik', 524: 'kitap',
    526: 'kitap', 528: 'muzik', 530: 'muzik'
}

# Kodda kullanılmıyor. Gerekirse kategorilere ağırlık verilebilir
category_weights = {
    'giyim': 1.0,       # Giyim ürünleri için temel ağırlık
    'kitap': 1.0,       # Kitaplar biraz daha ağırlıklı
    'elektronik': 1.0,  # Elektronik ürünler daha pahalı olabilir
    'mutfak': 1.0,      # Mutfak ürünleri daha düşük ağırlıklı
    'beyazEsya': 1.0,   # Beyaz eşya en yüksek ağırlığa sahip
    'aksesuar': 1.0,    # Aksesuarlar daha az ağırlıklı
    'kozmetik': 1.0,    # Kozmetik ürünler
    'muzik': 1.0,       # Müzik ürünleri
}

# 'category' sütununu ekleyerek veriyi dönüştürme
orders_df['category'] = orders_df['product_id'].map(category_mapping)

# Kategorileri ve ürün sayısını içeren yeni bir DataFrame oluşturma
weighted_orders = pd.DataFrame(index=orders_df['customer_id'].unique())

# Her kategori için ağırlıklı puanları hesaplama ve yeni DataFrame'e ekleme (hepsi için 1 category_weights)
for category, weight in category_weights.items():
    # Kategoriye göre filtreleme ve ağırlıklı toplamı hesaplama
    category_orders = orders_df[orders_df['category'] == category]
    weighted_totals = category_orders.groupby('customer_id')['product_qty'].sum() * weight
    weighted_orders[category] = weighted_totals

# NaN değerleri 0 ile doldurma
weighted_orders.fillna(0, inplace=True)

# Ağırlıklı sipariş DataFrame'ini göster
print(weighted_orders.head())

# Elbow Yöntemi ve Silhouette Skoru ile optimal küme sayısını belirleme
inertia = []
silhouette_scores = []
K = range(2, 11)  # En az 2 küme olması gerektiği için 2'den başlıyoruz

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42)
    labels = kmeanModel.fit_predict(weighted_orders)
    inertia.append(kmeanModel.inertia_)
    silhouette_scores.append(silhouette_score(weighted_orders, labels))

# Inertia değerlerindeki değişiklikleri hesaplama ve en büyük değişikliğin olduğu noktayı bulma
inertia_diff = np.diff(inertia)
knee_point = np.argmax(inertia_diff) + 2  # +2, çünkü aralık 2'den başlıyor ve 1. dereceden fark

# Elbow grafiğini çizme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # 1 satır, 2 sütun, 1. grafik
plt.plot(K, inertia, 'bx-')
plt.xlabel('Küme sayısı (k)')
plt.ylabel('Inertia')
plt.title('Elbow Yöntemi ile Küme Sayısı Belirleme')
plt.axvline(x=knee_point, color='red', linestyle='--')  # En iyi k sayısını işaretle

# Silhouette skorlarını görselleştirme
plt.subplot(1, 2, 2)  # 1 satır, 2 sütun, 2. grafik
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Küme sayısı (k)')
plt.ylabel('Silhouette Skoru')
plt.title('Silhouette Metodu ile Küme Sayısı Belirleme')

plt.tight_layout()
plt.show()

# En iyi silhouette skoruna sahip küme sayısını bul
best_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f"En iyi Silhouette Skoru {best_k} küme sayısı için: {max(silhouette_scores)}")

# K-Means modelini optimal küme sayısı ile eğitme
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(weighted_orders)  # Burada da ağırlıklı verileri kullanıyoruz

# PCA ile boyut indirgeme
pca = PCA(n_components=2)
weighted_orders_pca = pca.fit_transform(weighted_orders)  # Ağırlıklı verileri PCA'ya uyguluyoruz

# Küme etiketlerini ve merkezlerini çizme
plt.figure(figsize=(10, 8))
for i in range(knee_point):
    plt.scatter(weighted_orders_pca[kmeans.labels_ == i, 0], weighted_orders_pca[kmeans.labels_ == i, 1], label=f'Cluster {i+1}')
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# PCA ile 3 boyutlu indirgeme
pca = PCA(n_components=3)
weighted_orders_pca = pca.fit_transform(weighted_orders)

# 3D scatter plot çizme
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Her küme için noktaları çizme
for i in range(kmeans.n_clusters):
    ax.scatter(weighted_orders_pca[kmeans.labels_ == i, 0],
               weighted_orders_pca[kmeans.labels_ == i, 1],
               weighted_orders_pca[kmeans.labels_ == i, 2],
               label=f'Cluster {i+1}')

# Küme merkezlerini çizme
centers = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=300, c='yellow', label='Centroids')

ax.set_title('3D visualization of K-Means Clustering')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()

plt.show()

# PyVista plotter'ını başlat
plotter = pv.Plotter()

# Kümeler için renkleri tanımla
colors = [
    '#1f77b4',  # mavi
    '#ff7f0e',  # turuncu
    '#2ca02c',  # yeşil
    '#d62728',  # kırmızı
    '#9467bd',  # mor
    '#8c564b',  # kahverengi
    '#e377c2',  # pembe
    '#7f7f7f',  # gri
    '#bcbd22',  # lime yeşili
    '#17becf',  # açık mavi
    '#aec7e8',  # soluk mavi
    '#ffbb78',  # soluk turuncu
    '#98df8a',  # soluk yeşil
    '#ff9896',  # soluk kırmızı
    '#c5b0d5',  # soluk mor
    # Daha fazla renk eklenebilir
]

# Küme merkezleri için renk
centroid_color = '#000000'  # altın sarısı

# Her küme için noktaları çiz
for i in range(kmeans.n_clusters):
    cloud = pv.PolyData(weighted_orders_pca[kmeans.labels_ == i])
    plotter.add_mesh(cloud, color=colors[i], point_size=5.0, label=f'Cluster {i+1}')

# Küme merkezlerini çiz
centers = pca.transform(kmeans.cluster_centers_)
center_cloud = pv.PolyData(centers)
plotter.add_mesh(center_cloud, color=centroid_color, point_size=10.0, label='Centroids')

# Eksenleri çiz
plotter.show_axes()

# 3D görünümü başlatma (PyVista)
plotter.add_legend()
plotter.show()


# K-Means küme etiketlerini DataFrame'e ekleme
weighted_orders['Cluster'] = kmeans.labels_

# Her küme için kategori bazında ortalama alışveriş miktarlarını hesaplama
cluster_means = weighted_orders.groupby('Cluster').mean()

# Her küme için kategori bazında toplam alışveriş miktarlarını hesaplama
cluster_sums = weighted_orders.groupby('Cluster').sum()

# Ortalamaları ve toplamları CSV dosyalarına yazma, virgülden sonra 4 hane ile
cluster_means.to_csv('cluster_means.csv', float_format='%.4f')
cluster_sums.to_csv('cluster_sums.csv', float_format='%.0f')

# Her müşterinin ait olduğu küme etiketlerini ayrı bir DataFrame'e aktarma
customer_clusters = pd.DataFrame(weighted_orders['Cluster'])
customer_clusters.to_csv('customer_clusters.csv', float_format='%.4f')

print("Küme analizi dosyalara kaydedildi: 'cluster_means.csv', 'cluster_sums.csv'")
print("Müşteri küme etiketleri dosyaya kaydedildi: 'customer_clusters.csv'")


# Müşteri-cluster eşleştirmelerini içeren DataFrame'i yükleme
customer_clusters = pd.read_csv('customer_clusters.csv', index_col=0)

# Cluster ortalamalarını içeren DataFrame'i yükleme
cluster_means = pd.read_csv('cluster_means.csv', index_col=0)

# Her müşteri için en düşük ortalama değere sahip kategoriyi bulma
def en_dusuk_kategori_bul(cluster_id):
    return cluster_means.loc[cluster_id].idxmin()

# Her cluster için en çok ilgi gösterilen kategoriyi bulma
def en_cok_ilgi_gosterilen_kategori(cluster_id):
    return cluster_means.loc[cluster_id].idxmax()

# Müşterilerin en az ve en çok ilişkili oldukları kategorileri içeren yeni sütunları oluşturma
customer_clusters['En_Cok_Iliskili_Kategori'] = customer_clusters['Cluster'].apply(en_cok_ilgi_gosterilen_kategori)
customer_clusters['En_Az_Iliskili_Kategori'] = customer_clusters['Cluster'].apply(en_dusuk_kategori_bul)

# Sonuçları CSV dosyasına kaydetme
customer_clusters.to_csv('categories_by_cluster_means.csv')

print("Müşterilerin en az ilişkili oldukları kategoriler kaydedildi: 'categories_by_cluster_means.csv'")
