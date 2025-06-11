import streamlit as st
from keras.models import model_from_json
from pathlib import Path
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications import vgg16


# Load the json file that contains the model's structure
f = Path("model_structure.json")
model_structure = f.read_text()

model = model_from_json(model_structure)
model.load_weights("model.weights.h5")

st.title("หน้าคล้ายดาราคนไหน? 👀🎬")
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)


if picture:
    img = image.load_img(picture, target_size=(224, 224))
    image_array = image.img_to_array(img)
    images = np.expand_dims(image_array, axis=0)
    images = vgg16.preprocess_input(images)

    feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = feature_extraction_model.predict(images)

    # Given the extracted features, make a final prediction using our own model
    results = model.predict(features)
    print('Probability:', results)

    predicted_class = np.argmax(results)

    predicted_name = 'None'
    if predicted_class==0:
        predicted_name = 'ชมพู่ อารียา'
    elif predicted_class==1:
        predicted_name = 'ลิซ่า'
    elif predicted_class==2:
        predicted_name = 'ใหม่ ดาวิกา'    
    elif predicted_class==3:
        predicted_name = 'ตูน บอดีสแลม'
    elif predicted_class==4:
        predicted_name = 'หม่ำ จ๊กมก'   
    else:
        predicted_name = 'มาริโอ้'
    #print('This is',predicted_name,'with confidence:',results[0][np.argmax(results)]*100)


if st.button("ทดสอบ"):
    st.info(f"คุณหน้าคล้ายกับ: **{predicted_name}** 😲")

        