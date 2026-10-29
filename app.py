import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")

st.title("葡萄酒品质分类预测")
st.write("输入葡萄酒各项指标，预测品质等级")

alcohol = st.number_input("酒精度", value=10.50)
malic_acid = st.number_input("苹果酸", value=1.50)
ash = st.number_input("灰分", value=2.40)
alcalinity_of_ash = st.number_input("灰分碱度", value=15.00)
magnesium = st.number_input("镁", value=90.00)
total_phenols = st.number_input("总酚", value=2.20)
flavanoids = st.number_input("类黄酮", value=2.80)
nonflavanoid_phenols = st.number_input("非类黄酮酚", value=0.30)
proanthocyanins = st.number_input("原花青素", value=1.80)
color_intensity = st.number_input("颜色强度", value=5.00)
hue = st.number_input("色调", value=1.00)
od280 = st.number_input("稀释吸光度", value=3.00)
proline = st.number_input("脯氨酸", value=800.00)

if st.button("开始预测"):
    input_data = pd.DataFrame([[
        alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
        total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
        color_intensity, hue, od280, proline
    ]])
    pred = model.predict(input_data)[0]
    st.success(f"预测分类标签：{pred}")
