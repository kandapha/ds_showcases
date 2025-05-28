import streamlit as st


st.set_page_config(page_title="สาขาวิทยาการข้อมูล DS - TNI", layout="centered")

# Header
#st.title("🎓 วิทยาการข้อมูลและการวิเคราะห์ข้อมูล (Data Science)")
st.markdown("## 🎓 วิทยาการข้อมูลและการวิเคราะห์ข้อมูล Data Science and Data Analytics")

st.divider()

# ตัวอย่าง Application
st.markdown("#### 🤖 ตัวอย่าง AI Application")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### Weather")
    st.markdown(
    """
    <a href="/predict_weather">
        <img src="https://assets.thehansindia.com/h-upload/2019/12/01/241086-weather-forecast-andhra-pra.webp" width="200">
    </a>
    """,
    unsafe_allow_html=True
    )


with col2:
    st.markdown("#### Stock Price")
    st.markdown(
    """
    <a href="/horoscope">
        <img src="https://daxg39y63pxwu.cloudfront.net/images/blog/stock-price-prediction-using-machine-learning-project/Stock_Price_Prediction.webp" width="200">
    </a>
    """,
    unsafe_allow_html=True
    )


with col3:
    st.markdown("#### Your Next GPA")
    st.markdown(
    """
    <a href="/predict_your_next_GPA">
        <img src="https://www.shutterstock.com/image-photo/wooden-block-letters-gpa-word-260nw-2503751947.jpg" width="180">
    </a>
    """,
    unsafe_allow_html=True
    )


# จุดเด่นของหลักสูตร
st.divider()
st.markdown("#### 🔍 จุดเด่นของหลักสูตร DS")
st.markdown(""" - เรียนรู้การใช้ข้อมูลจริงในการแก้ปัญหา
- ใช้ AI และ Machine Learning เพื่อทำนายและตัดสินใจ
- ฝึกปฏิบัติผ่านโปรเจกต์ และเครื่องมือจริง เช่น Python, RapidMiner, และ Cloud Platform
- เรียนรู้การวิเคราะห์ Big Data, การทำ Visualization และการเขียนโปรแกรม
- มีเกมธุรกิจ (Scale Up! Game) ให้เข้าใจการจัดการกระบวนการในโลกจริง
""")

st.markdown("— [ดูรายละเอียดหลักสูตรเพิ่มเติม](https://www.tni.ac.th/it/major_ds) —")