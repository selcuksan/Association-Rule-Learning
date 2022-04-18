############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak


# !pip install mlxtend
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
from utils.aykiri_degisken_analizi import replace_with_thresholds

df_ = pd.read_excel("CRM Analitiği/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")


def retail_data_pred(dataframe):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    # 2. ARL Veri Yapısını hazırlama
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    # 3. Birliktelik kurallarının çıkarılması
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df = df_.copy()
df = retail_data_pred(df)

rules = create_rules(df)

rules[(rules["support"] > 0.05) & (rules["lift"] > 5) & (rules["confidence"] > 0.1)] \
    .sort_values("confidence", ascending=False)

# Sepet aşamasındaki kullanıcılara ürün önerisinde bulunmak
# Kullanıcı örnek ürün id: 22492
product_id = 22492
check_id(df, product_id)


def arl_recommender(rules_df, rec_count=1, sorting_param="lift"):
    sorted_rules = rules_df.sort_values(sorting_param, ascending=False)
    recommended_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        if product_id in list(product):
            recommended_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommended_list[0: rec_count]


arl_recommender(rules, 2, "confidence")
arl_recommender(rules, 2, "lift")
arl_recommender(rules, 2, "support")
