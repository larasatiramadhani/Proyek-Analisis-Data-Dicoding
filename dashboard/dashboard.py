import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



sns.set(style='dark')
st.header('Dicoding Collection :sparkles:')

def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    
    return daily_orders_df

def create_byreview_df(df):
    byreview_df = df.groupby(by="review_score").product_id.nunique().reset_index()
    byreview_df.rename(columns={
        "product_id": "product_count"
    }, inplace=True)
    
    return byreview_df

csv_url = 'https://docs.google.com/spreadsheets/d/15D40JfmXEFwwAFVXudHgq9Cka8Erk9--AN-WPFNCHbQ/export?format=csv&gid=1564547850'
all_df = pd.read_csv(csv_url)

datetime_columns = ["order_purchase_timestamp", "order_delivered_carrier_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)
 
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    # Menambahkan logo perusahaan
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]

daily_orders_df = create_daily_orders_df(main_df)
byreview_df = create_byreview_df(main_df)

st.header("Laras's project on Dicoding Collection Dashboard :sparkles:")
          
st.subheader('Daily Orders')
col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "AUD", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
 
st.pyplot(fig)
 
fig, ax = plt.subplots(figsize=(20, 10))
colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(
    x="review_score", 
    y="product_count",
    data=byreview_df.sort_values(by="review_score", ascending=False),
    palette=colors,
    ax=ax
)
ax.set_title("Number of Product by Review", loc="center", fontsize=30)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
st.pyplot(fig)

st.subheader('Clustering')
X = all_df[['price', 'freight_value']]

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering menggunakan KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
all_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualisasi hasil clustering dengan Matplotlib
fig, ax = plt.subplots()
scatter = ax.scatter(all_df['price'], all_df['freight_value'], c=all_df['Cluster'], cmap='viridis')
plt.title('Clustering berdasarkan Price dan Freight Value')
plt.xlabel('Price')
plt.ylabel('Freight Value')
plt.colorbar(scatter)

# Tampilkan visualisasi di Streamlit
st.pyplot(fig)

st.subheader('Product with the higher sold and the least sold')
url = 'https://docs.google.com/spreadsheets/d/1MSHuclurFzm0IuNyCfdWDBgDvgUe9jSaTUkZnTLvc2s/export?format=csv&gid=1431532174'
product_sales = pd.read_csv(url)
# Langkah 1: Kelompokkan data berdasarkan product_id dan hitung jumlah penjualan
product_sales_count = product_sales.groupby('product_id')['quantity'].sum().reset_index()

# Langkah 2: Ambil top 5 product yang terjual paling banyak dan paling sedikit
top_5_most_sold = product_sales_count.nlargest(5, 'quantity')
top_5_least_sold = product_sales_count.nsmallest(5, 'quantity')

# Langkah 3: Membuat visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Visualisasi untuk top 5 produk terjual paling banyak
sns.barplot(x='quantity', y='product_id', data=top_5_most_sold, ax=axes[0], palette='Greens_d')
axes[0].set_title('Top 5 Most Sold Products')
axes[0].set_xlabel('Quantity Sold')
axes[0].set_ylabel('Product ID')

# Visualisasi untuk top 5 produk terjual paling sedikit
sns.barplot(x='quantity', y='product_id', data=top_5_least_sold, ax=axes[1], palette='Reds_d')
axes[1].set_title('Top 5 Least Sold Products')
axes[1].set_xlabel('Quantity Sold')
axes[1].set_ylabel('Product ID')

# Menyesuaikan layout agar tidak terpotong
plt.tight_layout()

# Langkah 4: Tampilkan visualisasi di Streamlit
st.pyplot(fig)


url2 = 'https://docs.google.com/spreadsheets/d/1jLf4tAlvXfsa23IaOPKgivOObJbWNki_gv52n3SO4_g/export?format=csv&gid=948433226'
purchase_time_df = pd.read_csv(url2)

# Mengonversi kolom order_purchase_timestamp menjadi datetime
purchase_time_df['order_purchase_timestamp'] = pd.to_datetime(purchase_time_df['order_purchase_timestamp'])

# Membuat kolom baru untuk bulan dan tahun berdasarkan timestamp
purchase_time_df['purchase_month'] = purchase_time_df['order_purchase_timestamp'].dt.to_period('M')

# Menghitung jumlah order per bulan
monthly_sales = purchase_time_df.groupby('purchase_month').size()

# Menampilkan hasil fluktuasi penjualan
st.subheader("Fluktuasi Penjualan per Bulan")

# Langkah 1: Membuat visualisasi menggunakan matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
monthly_sales.plot(kind='line', ax=ax, marker='o', color='b')
ax.set_title('Fluktuasi Penjualan per Bulan')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Order')
plt.xticks(rotation=45)  # Memutar label sumbu-x agar lebih mudah dibaca
plt.tight_layout()

# Langkah 2: Deploy visualisasi ke Streamlit
st.pyplot(fig)

# Langkah 1: Baca data dari Google Sheets
url3 = 'https://docs.google.com/spreadsheets/d/1pe4ios97MGqXs1vhuSi8ECnkWmIq1CaLkmarcbGkvfc/export?format=csv'
customer_df = pd.read_csv(url3)

# Langkah 2: Mengelompokkan data dan menghitung jumlah pembeli unik
top_customers = customer_df.groupby(['customer_city', 'customer_state']).customer_unique_id.nunique().reset_index()

# Memberi nama kolom untuk kejelasan
top_customers.columns = ['customer_city', 'customer_state', 'num_customers']

# Mengurutkan hasil berdasarkan jumlah pembeli terbanyak
top_customers_sorted = top_customers.sort_values(by='num_customers', ascending=False)

# Menampilkan 5 kota dengan pembeli terbanyak
top_5_cities = top_customers_sorted.head(5)

# Menampilkan subheader di aplikasi Streamlit
st.subheader('Top 5 Cities with Most Unique Customers')

# Langkah 3: Visualisasikan menggunakan seaborn
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='num_customers', y='customer_city', data=top_5_cities, palette='Blues_d', ax=ax)
ax.set_title('Top 5 Cities with Most Unique Customers')
ax.set_xlabel('Number of Unique Customers')
ax.set_ylabel('Customer City')

# Tampilkan visualisasi di Streamlit
st.pyplot(fig)

# Menampilkan tabel data di Streamlit untuk memverifikasi
st.dataframe(top_5_cities)

