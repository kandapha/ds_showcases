import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


st.title("Compare Closing Price Trend: Two Stocks")

# อัปโหลดไฟล์
uploaded_file1 = st.file_uploader("Upload First Stock CSV", type=["csv"], key="file1")
uploaded_file2 = st.file_uploader("Upload Second Stock CSV", type=["csv"], key="file2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    # อ่านไฟล์ทั้งสอง
    df1 = pd.read_csv(uploaded_file1)
    df2 = pd.read_csv(uploaded_file2)

    # แปลงวันที่และจัดเรียง
    df1["Date"] = pd.to_datetime(df1["Date"])
    df2["Date"] = pd.to_datetime(df2["Date"])
    df1 = df1.sort_values("Date")
    df2 = df2.sort_values("Date")

    # ตั้งชื่อหุ้น (หรือให้ผู้ใช้กรอก)
    stock1_name = st.text_input("Stock 1 Name", value="Stock A")
    stock2_name = st.text_input("Stock 2 Name", value="Stock B")

    # เตรียมข้อมูลแสดงผล
    merged = pd.merge(df1[["Date", "Close"]], df2[["Date", "Close"]],
                      on="Date", suffixes=(f"_{stock1_name}", f"_{stock2_name}"))

    merged = merged.set_index("Date")
    merged.columns = [stock1_name, stock2_name]

    st.subheader("Line Chart: Compare Closing Prices")
    st.line_chart(merged)