import streamlit as st
import os
import random
from PIL import Image
import pandas as pd

# ตั้งค่าชื่อโฟลเดอร์ภาพ
IMAGE_FOLDER = "Cards-jpg"  # ให้สร้างโฟลเดอร์ชื่อ Cards-jpg และใส่ภาพไว้ที่นั่น
all_images = [img for img in os.listdir(IMAGE_FOLDER) if img.lower().endswith(('.jpg'))]


st.title("🔮 ดูดวงด้วยตัวเอง ด้วย Data Science")

# สุ่มภาพ 16 ภาพ
if st.button("🔀 Randomize"):
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


#df = pd.read_csv('horo.csv')
file_path = "horo.txt"
df = pd.read_csv(file_path, sep='\t', header=None, names=["Name", "Description", "Filename"])

st.divider()
#st.markdown("#### คำทำนายดวงคุณวันนี้")
st.markdown('<div id="my-target"><h2>🎯 คำทำนายดวงคุณวันนี้!</h2></div>', unsafe_allow_html=True)

if clicked_image:
    #clicked_image = 'Cups09.jpg'

    st.markdown("""
        <script>
            document.getElementById("my-target").scrollIntoView({behavior: "smooth"});
        </script>
    """, unsafe_allow_html=True)

    st.image(Image.open(os.path.join(IMAGE_FOLDER, clicked_image)) , width=150)
    st.info(f"You clicked on: **{clicked_image}**") 
    st.info(f"คำทำนาย: วันนี้ {df[df['Filename']==clicked_image]['Description'].values[0]}")    

    
