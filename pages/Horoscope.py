import streamlit as st
import os
import random
from PIL import Image
import pandas as pd
from time import sleep

# ตั้งค่าชื่อโฟลเดอร์ภาพ
IMAGE_FOLDER = "Cards-jpg"  # ให้สร้างโฟลเดอร์ชื่อ Cards-jpg และใส่ภาพไว้ที่นั่น
all_images = [img for img in os.listdir(IMAGE_FOLDER) if img.lower().endswith(('.jpg'))]


st.title("🔮 ดูดวงด้วยตัวเอง ด้วย Data Science")

with st.expander("อ่านวิธีการเล่น"):
    st.write("""
        1. ให้คุณหลับตา ตั้งจิตอธิฐาน แล้วกดปุ่ม "🔀 กดเพื่อสลับไพ่ และอย่าลืมหลับตา ตั้งจิตอธิฐาน 🙈" เพื่อสลับไพ่
        2. จากนั้นให้คุณเลือกไพ่ 1 ใบ โดยการกดปุ่ม "เลือก 👇" ใต้ไพ่ที่คุณชอบที่สุด
        3. เมื่อคุณเลือกไพ่แล้ว ระบบจะทำการทำนายดวงของคุณในวันนี้
        4. คุณสามารถกดปุ่ม "คำเตือน ไม่ควรดูดวงบ่อยเกินวันละ 1 ครั้ง!" เพื่อรีเซ็ตและดูดวงใหม่ได้อีกครั้งในวันถัดไป
    """)  

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2196F3;
        color: white;
        border-radius:10px;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)
# สุ่มภาพ 16 ภาพ
if st.button("""🙈 หลับตา ตั้งจิตอธิฐาน  🔀 กดที่นี่เพื่อสลับไพ่  จากนั้น เลือกไพ่ 1 ใบ โดยการกดปุ่ม "เลือก 👇" ใต้ไพ่ที่คุณชอบ"""):
    random_images = random.sample(all_images, min(20, len(all_images)))
    st.session_state["images"] = random_images


          
else:
    if "images" not in st.session_state:
        st.session_state["images"] = random.sample(all_images, min(20, len(all_images)))

# ตัวแปรเก็บภาพที่ถูกคลิก
clicked_image = None

cols = st.columns(5)
for i, img_name in enumerate(st.session_state["images"]):
    col = cols[i % 5]
    
    with col:
        if st.button("เลือก 👇", key=img_name ):  # empty button over image
            clicked_image = img_name
           
        img = Image.open(os.path.join(IMAGE_FOLDER, 'CardBacks.jpg'))
        st.image(img, use_container_width=True)
        #sleep(0.2) 
        

#df = pd.read_csv('horo.csv')
file_path = "horo.txt"
df = pd.read_csv(file_path, sep='\t', header=None, names=["Name", "Description", "Filename"])



@st.dialog("Cast your fortune")
def fortune(clicked_image,item):
    st.markdown('<div id="my-target"><h2>🎯 คำทำนายดวงคุณวันนี้!</h2></div>', unsafe_allow_html=True)
    #st.info(f"You clicked on: **{clicked_image}**")     
    #st.divider()
    st.image(Image.open(os.path.join(IMAGE_FOLDER, clicked_image)) , width=150)
    st.write(f"ดวงวันนี้ {item} ")
    if st.button("คำเตือน ไม่ควรดูดวงบ่อยเกินวันละ 1 ครั้ง!"):
        st.rerun()

if clicked_image:
    #clicked_image = 'Cups09.jpg'

    st.markdown("""
        <script>
            document.getElementById("my-target").scrollIntoView({behavior: "smooth"});
        </script>
    """, unsafe_allow_html=True)

    #st.image(Image.open(os.path.join(IMAGE_FOLDER, clicked_image)) , width=150)
    #st.info(f"You clicked on: **{clicked_image}**") 
    #st.info(f"คำทำนาย: วันนี้ {df[df['Filename']==clicked_image]['Description'].values[0]}")    
    #st.toast("คุณเลือกไพ่แล้ว! เลื่อนไปด้านล่างเพื่ออ่านผลการทำนาย", icon="🎴")

    fortune(clicked_image, df[df['Filename']==clicked_image]['Description'].values[0])

    




