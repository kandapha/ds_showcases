import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Step 1: Train Program model
@st.cache_resource
def load_model_and_encoders():
    # สำหรับตัวอย่างนี้ เราจะฝึก Decision Tree ใหม่ (คุณสามารถเปลี่ยนเป็นโหลดไฟล์ได้)
    df = pd.read_csv("Student_Behavior.csv")

    df_cleaned = df[df['Gender'].isin(['ชาย','หญิง'])]
    df_cleaned = df_cleaned[df_cleaned['Program'].isin(['บริหารธุรกิจ',
                                                        'เทคโนโลยีสารสนเทศ (IT) หรือ วิทยาการคอมพิวเตอร์ (Computer Science)',
                                                        'Multimedia Technology (MT)',
                                                        'Business Information Technology (BI)',
                                                        'สื่อสารมวลชน Digital Communication (DC)',
                                                        'สารสนเทศและนวัตกรรมดิจิทัล',
                                                        'บริหารธุรกิจ ระบบสารสนเทศและนวัตกรรมดิจิทัล',
                                                        'Multimedia Technology (MT)','วิทยาศาสตร์'])]
    
    df_cleaned.replace('เทคโนโลยีสารสนเทศ (IT) หรือ วิทยาการคอมพิวเตอร์ (Computer Science)', 'เทคโนโลยีสารสนเทศ (IT)', inplace=True)
    #df_cleaned.replace('Business Information Technology (BI)', 'Other', inplace=True)
    #df_cleaned.replace('สื่อสารมวลชน Digital Communication (DC)', 'Other', inplace=True)
    #df_cleaned.replace('Multimedia Technology (MT)', 'Other', inplace=True)
    df_cleaned.replace('วิทยาศาสตร์', 'Other', inplace=True)
    df_cleaned.replace('สารสนเทศและนวัตกรรมดิจิทัล', 'Business Information Technology (BI)', inplace=True)
    df_cleaned.replace('บริหารธุรกิจ ระบบสารสนเทศและนวัตกรรมดิจิทัล', 'Business Information Technology (BI)', inplace=True)
    
    df_cleaned["Favorite Leisure Activity"].replace("ดู F1", "อื่นๆ", inplace=True)
    df_cleaned["Favorite Leisure Activity"].replace("ตกปลา", "อื่นๆ", inplace=True)
    df_cleaned["Favorite Leisure Activity"].replace("หาอะไรทำ", "อื่นๆ", inplace=True)
    df_cleaned["Favorite Leisure Activity"].replace("ชอบนอน", "นอน", inplace=True)

    df_cleaned.replace('เคยดรอปมากกว่า 3 วิชา', 'เคย', inplace=True)
    df_cleaned.replace('เคย 1 วิชา', 'เคย', inplace=True)

    #df_cleaned["Program/Faculty Admission Success"].replace("ไม่มีความคาดหวังในตอนแรก", "ไม่ใช่", inplace=True)

    columns_to_drop = [
        'Satisfaction with Academic Performance',
        'Program/Faculty Admission Success',
        'University Admission Success',
        #'Least Favorite Subject',
        #'Favorite Subject',
        #'GPA Trend',
        'Program',
        'Dropped Any Subject',
        'Goal Setting in Learning',
        'Self-Discipline in Studying',
        'Mobile Usage Behavior',
        'Peak Study Time',
        'Most Used App Daily',
        'Favorite Leisure Activity',
        'Overall Life Satisfaction'
    ]

    df_cleaned.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df_cleaned.dropna(subset=['GPA Trend'], inplace=True)
    df_cleaned.dropna(inplace=True)

    label_encoders = {}
    df_encoded = df_cleaned.copy()
    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded.drop(columns=['GPA Trend'])
    y = df_encoded['GPA Trend']
    #clf = DecisionTreeClassifier(random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #clf_acc = DecisionTreeClassifier(random_state=42)
    clf_acc = LogisticRegression(max_iter=1000)
    clf_acc.fit(X_train, y_train)
    y_pred = clf_acc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, label_encoders, acc



# Step 2: Predict 
clf, label_encoders, acc_score = load_model_and_encoders()

# ----------------------------
# INTERFACE
# ----------------------------
st.title("📊 ทำนายอนาคตด้วย Data Science")

target = st.selectbox(
    label="คณะ/สาขาที่เรียนอยู่ปัจจุบัน",
    options=["Information Technology (IT)",
             "Data Science and Data Analytics (DS)",
             "Business Information Technology (BI)",
             "Multimedia Technology (MT)",
             "สื่อสารมวลชน Digital Communication (DC)", 
             "บริหาร", "วิศวกรรม","อื่นๆ"]
)

birth_month = st.selectbox(
    label="เดือนที่เกิด",
    options=["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
             "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
)

# รับข้อมูลจากผู้ใช้
input_data = {}
input_data_tmp = {}
input_data_tmp["Birth Month"] = birth_month

col1, col2 = st.columns(2)

for i, col in enumerate(clf.feature_names_in_):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        input_data[col] = st.selectbox(
            label=col,
            options=list(label_encoders[col].classes_)
    )
    

# เมื่อกดปุ่ม "ทำนาย"
if st.button("🔮 ทำนายผลการสอบ"):
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    prediction = clf.predict(input_df)[0]
    result = label_encoders['GPA Trend'].inverse_transform([prediction])[0]

    if result == 'เพิ่มขึ้น':
        st.success(f"📈 ยินดีด้วย โมเดลทำนายว่าน้องมีโอกาสจะได้เกรดเพิ่มขึ้น")
        st.metric(label="📊 โมเดลทำนายจากข้อมูลนักศึกษา 100 คน ความแม่นยำของโมเดล (Accuracy)", value=f"{acc_score * 100:.2f}%")

    elif result == 'ลดลง':
        st.error(f"📈 เสียใจด้วยนะ โมเดลทำนายว่าน้องมีโอกาสจะได้เกรดลดลง ตั้งใจขึ้นอีกนิดนะ")
        st.metric(label="📊 โมเดลทำนายจากข้อมูลนักศึกษา 100 คน ความแม่นยำของโมเดล (Accuracy)", value=f"{acc_score * 100:.2f}%")

    else:
        st.warning(f"📈 ตั้งใจอีกนิด น้องมีโอกาสจะได้เกรดเพิ่มขึ้นอย่างแน่นอน")
        st.metric(label="📊 โมเดลทำนายจากข้อมูลนักศึกษา 100 คน ความแม่นยำของโมเดล (Accuracy)", value=f"{acc_score * 100:.2f}%")



#if st.button("แสดงกฎการทำนาย"):
#    from sklearn.tree import export_text

#    tree_rules = export_text(clf, feature_names=list(clf.feature_names_in_))
#    st.code(tree_rules, language='text')  # แสดงในกล่อง code ใน Streamlit



