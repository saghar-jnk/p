import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('data/Diabetes Classification.csv')

df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
df['Gender'] = df['Gender'].replace({'FEMALE':'F','MALE':'M'})

df['LDL_to_HDL'] = df['LDL'] / df['HDL']
df['TG_to_HDL'] = df['TG'] / df['HDL']

def weighted_score(row, weights):
    num = den = 0
    for feat, w in weights.items():
        if feat in row and not pd.isna(row[feat]):
            num += row[feat] * w
            den += w
    return num / den

ckd_weights = {'Cr':0.45,'BUN':0.35,'BMI':0.15,'Age':0.05}
df['CKD_Risk_Score'] = df.apply(lambda r: weighted_score(r, ckd_weights), axis=1)

df['Fat_Index'] = 0.4*df['LDL_to_HDL'] + 0.3*df['TG_to_HDL'] + 0.3*df['BMI']
diabet_weights = {'Fat_Index':0.4,'BMI':0.3,'TG_to_HDL':0.3,'Age':0.1}
df['Diabetes_Risk'] = df.apply(lambda r: weighted_score(r, diabet_weights), axis=1)

cardio_weights = {'LDL_to_HDL':0.4,'TG_to_HDL':0.3,'Chol':0.2,'Age':0.1}
df['Cardio_Risk'] = df.apply(lambda r: weighted_score(r, cardio_weights), axis=1)

def score_to_level(series):
    try:
        return pd.qcut(series.rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
    except ValueError:
        bins = [series.min() - 1, series.quantile(0.33), series.quantile(0.66), series.max() + 1]
        return pd.cut(series, bins=bins, labels=['Low', 'Medium', 'High'])


df['CKD_Risk_Level'] = score_to_level(df['CKD_Risk_Score']).astype(str)
df['Diabetes_Risk_Level'] = score_to_level(df['Diabetes_Risk']).astype(str)
df['Cardio_Risk_Level'] = score_to_level(df['Cardio_Risk']).astype(str)

for risk in ['CKD_Risk_Level','Diabetes_Risk_Level','Cardio_Risk_Level']:
    print(f"\n{risk} by Gender (%):")
    print(
        df.groupby('Gender')[risk]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str) + '%'
    )

print("\n تمام شد.")

for col in ["CKD_Risk_Level", "Diabetes_Risk_Level", "Cardio_Risk_Level"]:
    df[col] = df[col].astype(str).str.capitalize()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
def search_risk(query_text):
    query_text = query_text.lower()
    result = df.copy()

    #جنسیت
    if "مرد" in query_text:
        result = result[result["Gender"] == "M"]
    if "زن" in query_text:
        result = result[result["Gender"] == "F"]

    # وضعیت بیماری کلیوی
    if "بیمار" in query_text or "کلیه" in query_text:
        result = result[result["Diagnosis"] == 1]
    if "سالم" in query_text:
        result = result[result["Diagnosis"] == 0]

    # ریسک دیابت
    if "ریسک دیابت بالا" in query_text:
        result = result[result["Diabetes_Risk_Level"] == "High"]
    if "ریسک دیابت متوسط" in query_text:
        result = result[result["Diabetes_Risk_Level"] == "Medium"]
    if "ریسک دیابت پایین" in query_text:
        result = result[result["Diabetes_Risk_Level"] == "Low"]

    # ریسک قلبی
    if "ریسک قلبی بالا" in query_text:
        result = result[result["Cardio_Risk_Level"] == "High"]
    if "ریسک قلبی متوسط" in query_text:
        result = result[result["Cardio_Risk_Level"] == "Medium"]
    if "ریسک قلبی پایین" in query_text:
        result = result[result["Cardio_Risk_Level"] == "Low"]

    # ریسک CKD
    if "ریسک کلیه بالا" in query_text:
        result = result[result["CKD_Risk_Level"] == "High"]
    if "ریسک کلیه متوسط" in query_text:
        result = result[result["CKD_Risk_Level"] == "Medium"]
    if "ریسک کلیه پایین" in query_text:
        result = result[result["CKD_Risk_Level"] == "Low"]


    print(f"تعداد رکوردهای پیدا شده: {len(result)}")
    return result


query = input("جمله خود را برای جستجو وارد کنید: ")
print(search_risk(query))



