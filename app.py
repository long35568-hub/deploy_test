import streamlit as st
import pandas as pd
import numpy as np
import re
import sqlite3
from datetime import datetime
from pathlib import Path
import os
import sys
import io
import json
import base64
import requests
import hashlib

# Thử import joblib, nếu không có thì dùng pickle
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False
    st.warning("⚠️ Joblib không có sẵn, đang dùng pickle. Cài đặt: pip install joblib")

# Cấu hình trang
st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="📧",
    layout="wide"
)

# ------------------------------------------------------------
# GitHub sync helpers
# ------------------------------------------------------------
def _compute_file_hash(path: str) -> str:
    """Tính MD5 hash nhanh cho file (dùng để kiểm tra thay đổi)."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return ""

def upload_db_to_github():
    """
    Đồng bộ file emails.db lên GitHub repo sử dụng PAT lưu trong st.secrets.
    Yêu cầu trong Streamlit Secrets có: GITHUB_TOKEN, GITHUB_USERNAME, GITHUB_REPO, GITHUB_FILE
    """
    db_path = "emails.db"
    if not os.path.exists(db_path):
        st.info("ℹ️ Không tìm thấy emails.db để đồng bộ.")
        return False

    # Kiểm tra secrets
    try:
        token = st.secrets["GITHUB_TOKEN"]
        username = st.secrets["GITHUB_USERNAME"]
        repo = st.secrets["GITHUB_REPO"]
        filename = st.secrets.get("GITHUB_FILE", "emails.db")
    except Exception as e:
        st.warning("⚠️ GitHub secrets chưa cấu hình. Bỏ qua đồng bộ lên GitHub.")
        return False

    # Tránh push nếu file không đổi (session cache)
    current_hash = _compute_file_hash(db_path)
    last_hash = st.session_state.get("github_last_db_hash")
    if last_hash == current_hash:
        # không thay đổi
        return True

    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{filename}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    # đọc file và encode base64
    with open(db_path, "rb") as f:
        content_bytes = f.read()
    encoded = base64.b64encode(content_bytes).decode()

    # Lấy SHA hiện tại (nếu file đã tồn tại trên repo)
    try:
        r = requests.get(api_url, headers=headers, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
        elif r.status_code == 404:
            sha = None
        else:
            st.warning(f"⚠️ Lỗi khi kiểm tra file trên GitHub: {r.status_code}")
            sha = None
    except Exception as e:
        st.warning(f"⚠️ Lỗi kết nối GitHub: {str(e)}")
        return False

    payload = {
        "message": f"Auto-update emails.db from Streamlit at {datetime.utcnow().isoformat()}Z",
        "content": encoded
    }
    if sha:
        payload["sha"] = sha

    try:
        put = requests.put(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        if put.status_code in (200, 201):
            st.session_state["github_last_db_hash"] = current_hash
            # Thông báo nhẹ nhàng (dùng st.success để hiện tại)
            st.success("📤 Database đã được đồng bộ lên GitHub.")
            return True
        else:
            # Show response message for debugging
            try:
                err = put.json()
            except:
                err = put.text
            st.warning(f"⚠️ Không thể push lên GitHub ({put.status_code}): {err}")
            return False
    except Exception as e:
        st.warning(f"⚠️ Lỗi khi push lên GitHub: {str(e)}")
        return False

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .email-form {
        background-color: #f0f2f6;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .spam {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .ham {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
    }
    
    .model-info {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
        color: #1565c0;
    }
    .model-info h4 {
        color: #0d47a1;
        margin-bottom: 10px;
    }
    .model-info p {
        color: #1976d2;
        margin: 5px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .analysis-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #6c757d;
        margin: 10px 0;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Database functions
# ------------------------------------------------------------
def init_database():
    """Khởi tạo SQLite database"""
    conn = sqlite3.connect('emails.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS emails
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  from_email TEXT,
                  to_email TEXT,
                  subject TEXT,
                  content TEXT,
                  prediction TEXT,
                  confidence TEXT)''')
    conn.commit()
    return conn

def load_emails_from_db(conn):
    """Load emails từ database"""
    try:
        df = pd.read_sql_query("SELECT * FROM emails ORDER BY timestamp DESC", conn)
        if len(df) > 0:
            # Rename columns để match với format cũ
            df = df.rename(columns={
                'from_email': 'From',
                'to_email': 'To',
                'subject': 'Subject',
                'content': 'Content',
                'prediction': 'Prediction',
                'confidence': 'Confidence',
                'timestamp': 'Timestamp'
            })
        return df
    except Exception:
        return pd.DataFrame(columns=['Timestamp', 'From', 'To', 'Subject', 'Content', 'Prediction', 'Confidence'])

def save_email_to_db(conn, email_data):
    """Lưu email vào database"""
    c = conn.cursor()
    c.execute('''INSERT INTO emails (timestamp, from_email, to_email, subject, content, prediction, confidence)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (email_data['Timestamp'], email_data['From'], email_data['To'], 
               email_data['Subject'], email_data['Content'], 
               email_data['Prediction'], email_data['Confidence']))
    conn.commit()

def clear_database(conn):
    """Xóa toàn bộ database"""
    c = conn.cursor()
    c.execute("DELETE FROM emails")
    conn.commit()

def create_csv_template():
    """Tạo template CSV mẫu"""
    template_data = {
        'From': ['sender1@example.com', 'sender2@example.com', 'promotion@shop.com'],
        'To': ['recipient@example.com', 'recipient@example.com', 'recipient@example.com'],
        'Subject': [
            'Meeting reminder for tomorrow',
            'Your order has been shipped',
            'CONGRATULATIONS! You won $1,000,000'
        ],
        'Content': [
            'Hi, this is a reminder about our meeting tomorrow at 2pm. Please bring your reports.',
            'Your order #12345 has been shipped and will arrive in 3-5 business days.',
            'Click here NOW to claim your prize! Limited time offer! Act fast or lose your winnings!'
        ]
    }
    template_df = pd.DataFrame(template_data)
    return template_df

# Khởi tạo database connection
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_database()

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Load model Decision Tree với joblib"""
    model_path = 'model.mdl'
    
    if not os.path.exists(model_path):
        return None, None, f"❌ File '{model_path}' không tồn tại trong thư mục: {os.getcwd()}"
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        return None, None, f"❌ File '{model_path}' rỗng (0 bytes)"
    
    try:
        if USE_JOBLIB:
            try:
                with st.spinner(f'⏳ Đang load model với joblib... ({file_size/1024:.1f} KB)'):
                    model = joblib.load(model_path)
                load_method = "joblib"
            except Exception as joblib_error:
                st.info("🔄 Đang thử load với pickle (cross-version mode)...")
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                load_method = "pickle (latin1 encoding)"
        else:
            import pickle
            with st.spinner(f'⏳ Đang load model với pickle... ({file_size/1024:.1f} KB)'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
            load_method = "pickle (latin1 encoding)"
        
        # Lấy feature names
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        elif hasattr(model, 'feature_names_'):
            feature_names = model.feature_names_.tolist()
        elif hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            try:
                test_input = np.zeros((1, 3000))
                model.predict(test_input)
                feature_names = [f"feature_{i}" for i in range(3000)]
            except:
                feature_names = None
        
        return model, feature_names, f"✅ Model loaded với {load_method}! (Size: {file_size/1024:.1f} KB)"
        
    except Exception as e:
        return None, None, f"❌ Lỗi: {str(e)[:300]}"

def extract_features_from_email(email_text, feature_names):
    """Trích xuất features từ email"""
    if feature_names is None:
        return None
    
    try:
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', email_text)
        words = clean_text.lower().split()
        
        email_vector = np.zeros(len(feature_names), dtype=np.int32)
        
        for i, feature in enumerate(feature_names):
            count = words.count(feature.lower())
            email_vector[i] = count
        
        return email_vector
    except Exception as e:
        st.error(f"Lỗi extract features: {str(e)}")
        return None

def predict_with_trained_model(email_text, model, feature_names):
    """Dự đoán với model đã train"""
    try:
        features = extract_features_from_email(email_text, feature_names)
        
        if features is None:
            return None, None, None
        
        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped)[0]
        
        # Lấy confidence
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_reshaped)[0]
            # proba is array indexed by class label order: find index of predicted class
            try:
                pred_index = list(model.classes_).index(prediction)
                confidence = proba[pred_index] * 100
            except Exception:
                # fallback: max probability
                confidence = proba.max() * 100
        
        return int(prediction), confidence, features
        
    except Exception as e:
        st.error(f"Lỗi dự đoán: {str(e)}")
        return None, None, None

def analyze_prediction(email_text, features, feature_names, prediction):
    """Phân tích tại sao email được phân loại như vậy"""
    if features is None or feature_names is None:
        return "Không thể phân tích"
    
    # Tìm top keywords có trong email
    non_zero_indices = np.nonzero(features)[0]
    
    if len(non_zero_indices) == 0:
        return "Email không chứa từ khóa nào trong training data."
    
    # Lấy top 15 từ xuất hiện nhiều nhất
    top_indices = non_zero_indices[np.argsort(features[non_zero_indices])[::-1][:15]]
    top_keywords = [(feature_names[i], int(features[i])) for i in top_indices]
    
    analysis = f"**Email {'SPAM' if prediction == 1 else 'HAM'} vì:**\n\n"
    
    if prediction == 1:
        analysis += "🚨 **Các dấu hiệu spam phát hiện:**\n\n"
    else:
        analysis += "✅ **Các dấu hiệu email thông thường:**\n\n"
    
    analysis += "**Top từ khóa xuất hiện trong email:**\n"
    for keyword, count in top_keywords:
        analysis += f"- `{keyword}`: xuất hiện {count} lần\n"
    
    # Phân tích cấu trúc
    word_count = len(email_text.split())
    upper_count = sum(1 for c in email_text if c.isupper())
    exclamation_count = email_text.count('!')
    dollar_count = email_text.count('$')
    url_count = len(re.findall(r'http[s]?://', email_text))
    
    analysis += f"\n**Đặc điểm cấu trúc:**\n"
    analysis += f"- Tổng số từ: {word_count}\n"
    analysis += f"- Chữ in hoa: {upper_count} ({upper_count/len(email_text)*100:.1f}%)\n"
    analysis += f"- Dấu chấm than (!): {exclamation_count}\n"
    analysis += f"- Ký tự đô la ($): {dollar_count}\n"
    analysis += f"- URL links: {url_count}\n"
    
    return analysis

# Load model
with st.spinner('⏳ Đang load model...'):
    model, feature_names, load_message = load_trained_model()

# Header
st.markdown('<div class="main-header"><h1>📧 Hệ Thống Phát Hiện Email Rác</h1><p>Mail Server Simulation & Spam Detection</p></div>', unsafe_allow_html=True)

# Hiển thị trạng thái model
if model is not None:
    st.markdown(f"""
    <div class="model-info">
        <h4>✅ Model hoạt động tốt!</h4>
        <p><strong>Model:</strong> Decision Tree Classifier</p>
        <p><strong>Features:</strong> {len(feature_names) if feature_names else 'N/A'} từ khóa</p>
        <p><strong>Model file:</strong> model.mdl</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error(f"❌ Không thể load model: {load_message}")
    st.stop()

# Tạo tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📨 Gửi Email", "📤 Upload CSV", "📊 Database", "📈 Thống Kê", "🧪 Test"])

# Tab 1: Gửi Email
with tab1:
    st.markdown('<div class="email-form">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✉️ Soạn Email Mới")
        
        with st.form("email_form"):
            from_email = st.text_input(
                "Từ (From):",
                placeholder="user@example.com",
                help="Địa chỉ email người gửi"
            )
            
            to_email = st.text_input(
                "Đến (To):",
                placeholder="recipient@example.com",
                help="Địa chỉ email người nhận"
            )
            
            subject = st.text_input(
                "Tiêu đề (Subject):",
                placeholder="Nhập tiêu đề email...",
                help="Tiêu đề của email"
            )
            
            content = st.text_area(
                "Nội dung (Content):",
                placeholder="Nhập nội dung email...",
                height=200,
                help="Nội dung chi tiết của email"
            )
            
            submit_button = st.form_submit_button("📤 Gửi Email", use_container_width=True)
        
        if submit_button:
            if not from_email or not to_email or not subject or not content:
                st.error("❌ Vui lòng điền đầy đủ thông tin!")
            else:
                full_text = f"{subject} {content}"
                
                prediction, confidence, features = predict_with_trained_model(full_text, model, feature_names)
                
                if prediction is not None:
                    # Lưu vào database
                    new_email = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'From': from_email,
                        'To': to_email,
                        'Subject': subject,
                        'Content': content,
                        'Prediction': 'SPAM' if prediction == 1 else 'HAM',
                        'Confidence': f"{confidence:.1f}%" if confidence else "N/A"
                    }
                    
                    save_email_to_db(st.session_state.db_conn, new_email)
                    
                    # Đồng bộ lên GitHub (nếu secrets có cấu hình)
                    upload_db_to_github()
                    
                    st.success("✅ Email đã được gửi và phân tích!")
                    
                    if prediction == 1:
                        st.markdown(
                            '<div class="result-box spam">🚨 CẢNH BÁO: Email này là THƯ RÁC (SPAM)</div>',
                            unsafe_allow_html=True
                        )
                        if confidence:
                            st.error(f"**Độ tin cậy:** {confidence:.1f}%")
                    else:
                        st.markdown(
                            '<div class="result-box ham">✅ Email này là THƯ THÔNG THƯỜNG (HAM)</div>',
                            unsafe_allow_html=True
                        )
                        if confidence:
                            st.success(f"**Độ tin cậy:** {confidence:.1f}%")
                else:
                    st.error("❌ Không thể phân tích email!")
    
    with col2:
        st.subheader("📖 Hướng dẫn")
        st.info("""
        **Cách sử dụng:**
        
        1. ✏️ Điền thông tin email
        2. 📤 Nhấn "Gửi Email"
        3. 🤖 AI phân tích tự động
        4. 👀 Xem kết quả chi tiết
        5. 💾 Dữ liệu được lưu vĩnh viễn
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Upload CSV
with tab2:
    st.subheader("📤 Upload và Phân Loại Email Hàng Loạt")
    
    # Thêm button tải template
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("""
        **Yêu cầu format CSV:**
        - Phải có cột: `Subject` (tiêu đề) và `Content` (nội dung)
        - Có thể có thêm: `From`, `To` (nếu không có sẽ để mặc định)
        """)
    
    with col2:
        # Tạo template và cho phép tải xuống
        template_df = create_csv_template()
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải Template CSV",
            data=template_csv,
            file_name="email_template.csv",
            mime="text/csv",
            help="Tải file mẫu để tham khảo format",
            use_container_width=True
        )
    
    # Preview template
    with st.expander("👁️ Xem cấu trúc template CSV"):
        st.dataframe(template_df, use_container_width=True)
        st.caption("File mẫu có 3 emails: 2 HAM và 1 SPAM để bạn tham khảo")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            st.write(f"**Đã tải lên:** {len(df_upload)} emails")
            st.dataframe(df_upload.head(), use_container_width=True)
            
            # Kiểm tra columns
            required_cols = ['Subject', 'Content']
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"❌ Thiếu cột bắt buộc: {', '.join(missing_cols)}")
            else:
                if st.button("🔍 Phân Loại Tất Cả", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for idx, row in df_upload.iterrows():
                        status_text.text(f"Đang phân tích email {idx+1}/{len(df_upload)}...")
                        progress_bar.progress((idx + 1) / len(df_upload))
                        
                        full_text = f"{row['Subject']} {row['Content']}"
                        prediction, confidence, _ = predict_with_trained_model(full_text, model, feature_names)
                        
                        # Lưu vào database
                        new_email = {
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'From': row.get('From', 'uploaded@csv.com'),
                            'To': row.get('To', 'system@example.com'),
                            'Subject': row['Subject'],
                            'Content': row['Content'],
                            'Prediction': 'SPAM' if prediction == 1 else 'HAM',
                            'Confidence': f"{confidence:.1f}%" if confidence else "N/A"
                        }
                        
                        save_email_to_db(st.session_state.db_conn, new_email)
                        
                        results.append({
                            'Subject': row['Subject'],
                            'Prediction': 'SPAM' if prediction == 1 else 'HAM',
                            'Confidence': f"{confidence:.1f}%" if confidence else "N/A"
                        })
                    
                    # Sau khi hoàn thành batch, đồng bộ 1 lần
                    upload_db_to_github()
                    
                    status_text.text("✅ Hoàn thành!")
                    
                    # Hiển thị kết quả
                    results_df = pd.DataFrame(results)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        spam_count = len(results_df[results_df['Prediction'] == 'SPAM'])
                        st.metric("🚨 SPAM", spam_count, f"{spam_count/len(results_df)*100:.1f}%")
                    
                    with col2:
                        ham_count = len(results_df[results_df['Prediction'] == 'HAM'])
                        st.metric("✅ HAM", ham_count, f"{ham_count/len(results_df)*100:.1f}%")
                    
                    with col3:
                        st.metric("📧 Tổng số", len(results_df))
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download kết quả
                    csv_result = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải Kết Quả CSV",
                        data=csv_result,
                        file_name=f"spam_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {str(e)}")

# Tab 3: Database
with tab3:
    st.subheader("📊 Email Database")
    
    # Load emails từ database
    emails_df = load_emails_from_db(st.session_state.db_conn)
    
    if len(emails_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_type = st.selectbox("Lọc theo loại:", ["Tất cả", "SPAM", "HAM"])
        
        with col2:
            sort_by = st.selectbox("Sắp xếp:", ["Mới nhất", "Cũ nhất"])
        
        with col3:
            search_term = st.text_input("🔍 Tìm kiếm:")
        
        with col4:
            st.metric("📧 Tổng số", len(emails_df))
        
        # Áp dụng filter
        filtered_df = emails_df.copy()
        
        if filter_type != "Tất cả":
            filtered_df = filtered_df[filtered_df['Prediction'] == filter_type]
        
        if search_term:
            mask = (
                filtered_df['Subject'].str.contains(search_term, case=False, na=False) |
                filtered_df['Content'].str.contains(search_term, case=False, na=False) |
                filtered_df['From'].str.contains(search_term, case=False, na=False) |
                filtered_df['To'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        
        if sort_by == "Mới nhất":
            filtered_df = filtered_df.sort_values('Timestamp', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('Timestamp', ascending=True)
        
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải CSV",
                data=csv,
                file_name=f"emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # SỬA LỖI: Thêm key unique và không dùng st.rerun()
            if st.button("🗑️ Xóa toàn bộ", key="delete_all_db", use_container_width=True):
                if st.session_state.get('confirm_delete', False):
                    clear_database(st.session_state.db_conn)
                    # Đồng bộ lên GitHub (xóa file trên repo sẽ được ghi mới - file rỗng DB)
                    upload_db_to_github()
                    st.session_state.confirm_delete = False
                    st.success("✅ Đã xóa toàn bộ database!")
                    # Không dùng st.rerun() để tránh quay về tab đầu tiên
                else:
                    st.session_state.confirm_delete = True
                    st.warning("⚠️ Nhấn lần nữa để xác nhận!")
    else:
        st.info("🔭 Chưa có email nào. Hãy gửi email đầu tiên!")

# Tab 4: Thống kê
with tab4:
    st.subheader("📈 Thống Kê & Phân Tích")
    
    emails_df = load_emails_from_db(st.session_state.db_conn)
    
    if len(emails_df) > 0:
        spam_count = len(emails_df[emails_df['Prediction'] == 'SPAM'])
        ham_count = len(emails_df[emails_df['Prediction'] == 'HAM'])
        total_count = len(emails_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚨 Thư rác", spam_count, f"{(spam_count/total_count*100):.1f}%")
        
        with col2:
            st.metric("✅ Thư thường", ham_count, f"{(ham_count/total_count*100):.1f}%")
        
        with col3:
            st.metric("📧 Tổng số", total_count)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Phân bố loại email")
            chart_data = pd.DataFrame({
                'Loại': ['SPAM', 'HAM'],
                'Số lượng': [spam_count, ham_count]
            })
            st.bar_chart(chart_data.set_index('Loại'))
        
        with col2:
            st.subheader("Email theo thời gian")
            try:
                time_df = emails_df.copy()
                time_df['Date'] = pd.to_datetime(time_df['Timestamp']).dt.date
                daily_count = time_df.groupby('Date').size().reset_index(name='Count')
                st.line_chart(daily_count.set_index('Date'))
            except:
                st.info("Chưa đủ dữ liệu")
    else:
        st.info("📊 Chưa có dữ liệu để hiển thị thống kê.")

# Tab 5: Test với phân tích chi tiết
with tab5:
    st.subheader("🧪 Test Email Spam Detection")
    
    test_email = st.text_area(
        "Nhập nội dung email để test:",
        placeholder="Nhập tiêu đề và nội dung email...",
        height=150
    )
    
    if st.button("🔍 Phân tích ngay", use_container_width=True):
        if test_email:
            prediction, confidence, features = predict_with_trained_model(test_email, model, feature_names)
            
            if prediction is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 SPAM")
                    else:
                        st.success("✅ HAM")
                
                with col2:
                    if confidence:
                        st.metric("Độ tin cậy", f"{confidence:.1f}%")
                
                # Hiển thị phân tích chi tiết
                st.markdown("---")
                st.subheader("📊 Phân Tích Chi Tiết")
                
                analysis = analyze_prediction(test_email, features, feature_names, prediction)
                
                st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
                
            else:
                st.error("❌ Không thể phân tích!")
        else:
            st.warning("⚠️ Vui lòng nhập nội dung!")
    
    st.markdown("---")
    st.markdown("### 📝 Ví dụ email")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**❌ SPAM:**")
        spam_ex = """CONGRATULATIONS!!! You WON $1,000,000!
Click here NOW! Limited time offer!
Verify account immediately or lose prize!"""
        st.code(spam_ex)
    
    with col2:
        st.markdown("**✅ HAM:**")
        ham_ex = """Meeting Reminder: Project Review
Hi team, reminder about tomorrow's meeting at 2pm.
Please bring your progress reports."""
        st.code(ham_ex)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📧 Email Spam Detection System | ML Model (Decision Tree)</p>
    <p>🐍 Python {sys.version_info.major}.{sys.version_info.minor} | 💾 Features: {len(feature_names) if feature_names else 'N/A'}</p>
</div>
""", unsafe_allow_html=True)
