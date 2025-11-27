import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
# import joblib
from pathlib import Path
import os

# Cấu hình trang
st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="📧",
    layout="wide"
)

# CSS cho giao diện đẹp hơn
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
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
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'emails_df' not in st.session_state:
    if os.path.exists('emails_database.xlsx'):
        try:
            st.session_state.emails_df = pd.read_excel('emails_database.xlsx')
        except:
            st.session_state.emails_df = pd.DataFrame(
                columns=['Timestamp', 'From', 'To', 'Subject', 'Content', 'Prediction']
            )
    else:
        st.session_state.emails_df = pd.DataFrame(
            columns=['Timestamp', 'From', 'To', 'Subject', 'Content', 'Prediction']
        )

# Hàm dự đoán email đơn giản (không cần model phức tạp)
def predict_email_simple(email_text):
    """
    Dự đoán email bằng thuật toán đơn giản dựa trên từ khóa
    Tiết kiệm bộ nhớ, không cần load model lớn
    """
    # Danh sách từ khóa spam phổ biến
    spam_keywords = {
        # Từ khóa về tiền bạc và quà tặng
        'free', 'win', 'winner', 'prize', 'cash', 'money', 'dollars', 
        'million', 'billion', 'reward', 'bonus', 'gift', 'earn',
        
        # Từ khóa khẩn cấp
        'urgent', 'immediately', 'act now', 'hurry', 'limited time',
        'expire', 'deadline', 'last chance', 'today only',
        
        # Từ khóa lừa đảo
        'congratulations', 'selected', 'claim', 'verify', 'confirm',
        'account', 'password', 'update', 'suspended', 'locked',
        
        # Từ khóa marketing
        'click here', 'click now', 'subscribe', 'unsubscribe',
        'offer', 'discount', 'cheap', 'lowest price', 'deal',
        
        # Từ khóa y tế/thuốc
        'viagra', 'cialis', 'pills', 'pharmacy', 'medication',
        'weight loss', 'lose weight',
        
        # Từ khóa ngân hàng/tài chính
        'credit card', 'loan', 'debt', 'investment', 'bitcoin',
        'forex', 'trading', 'casino', 'lottery'
    }
    
    # Chuyển text về lowercase
    text_lower = email_text.lower()
    
    # Đếm số từ khóa spam
    spam_score = 0
    found_keywords = []
    
    for keyword in spam_keywords:
        if keyword in text_lower:
            spam_score += 1
            found_keywords.append(keyword)
    
    # Kiểm tra các dấu hiệu khác
    # Nhiều dấu chấm than hoặc hỏi
    if text_lower.count('!') > 3 or text_lower.count('?') > 3:
        spam_score += 1
        
    # Chữ viết hoa nhiều
    if sum(1 for c in email_text if c.isupper()) > len(email_text) * 0.3:
        spam_score += 1
        
    # Nhiều số
    if sum(1 for c in email_text if c.isdigit()) > len(email_text) * 0.15:
        spam_score += 1
    
    # Quyết định: nếu spam_score >= 2 thì là spam
    is_spam = spam_score >= 2
    
    return {
        'prediction': 1 if is_spam else 0,
        'spam_score': spam_score,
        'found_keywords': found_keywords
    }

# Hàm load model từ file (tối ưu bộ nhớ)
@st.cache_resource
def load_model_optimized():
    """
    Load model đã train sẵn nếu có
    Sử dụng caching để tránh load nhiều lần
    """
    try:
        if os.path.exists('model_tree.pkl'):
            with st.spinner('⏳ Đang load model...'):
                model = joblib.load('model_tree.pkl')
                
                # Lấy feature names từ model
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_.tolist()
                else:
                    # Fallback: tạo danh sách feature mặc định
                    feature_names = None
                
                return model, feature_names
        else:
            return None, None
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {str(e)}")
        return None, None

# Hàm dự đoán với model ML (nếu có)
def predict_with_model(email_text, model, feature_names):
    """
    Dự đoán bằng model ML (tối ưu bộ nhớ)
    """
    try:
        if model is None or feature_names is None:
            return None
        
        # Xử lý văn bản
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', email_text)
        words = clean_text.lower().split()
        
        # Tạo vector frequency (chỉ tạo 1 sample)
        email_vector = np.zeros(len(feature_names), dtype=np.float32)  # Dùng float32 thay vì float64
        
        for i, feature in enumerate(feature_names):
            email_vector[i] = words.count(feature.lower())
        
        # Reshape thành 1 hàng
        email_vector = email_vector.reshape(1, -1)
        
        # Predict
        prediction = model.predict(email_vector)[0]
        
        return prediction
        
    except Exception as e:
        st.warning(f"⚠️ Lỗi khi dự đoán với model: {str(e)}")
        return None

# Load model (nếu có)
model, feature_names = load_model_optimized()

# Header
st.markdown('<div class="main-header"><h1>📧 Hệ Thống Phát Hiện Email Rác</h1><p>Mail Server Simulation & Spam Detection</p></div>', unsafe_allow_html=True)

# Hiển thị trạng thái model
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if model is not None:
        st.success("✅ Đang sử dụng Machine Learning Model")
    else:
        st.info("ℹ️ Đang sử dụng Rule-based Algorithm (không cần model)")

# Tạo tabs
tab1, tab2, tab3 = st.tabs(["📨 Gửi Email", "📊 Database", "📈 Thống Kê"])

# Tab 1: Gửi Email
with tab1:
    st.markdown('<div class="email-form">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✉️ Soạn Email Mới")
        
        # Form gửi email
        with st.form("email_form"):
            from_email = st.text_input(
                "Từ (From):",
                placeholder="your.email@example.com",
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
            
            submit_button = st.form_submit_button("📤 Gửi Email")
        
        if submit_button:
            if not from_email or not to_email or not subject or not content:
                st.error("❌ Vui lòng điền đầy đủ thông tin!")
            else:
                # Kết hợp tiêu đề và nội dung để phân tích
                full_text = f"{subject} {content}"
                
                # Dự đoán
                prediction = None
                spam_score = 0
                found_keywords = []
                
                # Thử dùng model ML trước
                if model is not None and feature_names is not None:
                    prediction = predict_with_model(full_text, model, feature_names)
                
                # Nếu model không hoạt động, dùng thuật toán đơn giản
                if prediction is None:
                    result = predict_email_simple(full_text)
                    prediction = result['prediction']
                    spam_score = result['spam_score']
                    found_keywords = result['found_keywords']
                
                # Lưu vào database
                new_email = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'From': from_email,
                    'To': to_email,
                    'Subject': subject,
                    'Content': content,
                    'Prediction': 'SPAM' if prediction == 1 else 'HAM'
                }
                
                # Thêm vào dataframe (append nhẹ hơn concat)
                new_row = pd.DataFrame([new_email])
                st.session_state.emails_df = pd.concat([
                    st.session_state.emails_df,
                    new_row
                ], ignore_index=True)
                
                # Lưu vào file Excel (chỉ lưu khi có thay đổi)
                try:
                    st.session_state.emails_df.to_excel('emails_database.xlsx', index=False)
                except:
                    st.warning("⚠️ Không thể lưu vào Excel, dữ liệu vẫn được lưu trong session")
                
                # Hiển thị kết quả
                st.success("✅ Email đã được gửi và phân tích!")
                
                if prediction == 1:
                    st.markdown(
                        '<div class="result-box spam">🚨 CẢNH BÁO: Email này được phát hiện là THƯ RÁC (SPAM)</div>',
                        unsafe_allow_html=True
                    )
                    if found_keywords:
                        st.warning(f"**Từ khóa spam phát hiện:** {', '.join(found_keywords)}")
                        st.info(f"**Điểm spam:** {spam_score}/10")
                else:
                    st.markdown(
                        '<div class="result-box ham">✅ Email này là THƯ THÔNG THƯỜNG (HAM) - An toàn</div>',
                        unsafe_allow_html=True
                    )
    
    with col2:
        st.subheader("📝 Hướng dẫn")
        st.info("""
        **Cách sử dụng:**
        
        1. Điền địa chỉ email người gửi
        2. Điền địa chỉ email người nhận
        3. Nhập tiêu đề email
        4. Nhập nội dung email
        5. Nhấn "Gửi Email"
        
        **Hệ thống sẽ:**
        - Phân tích nội dung email
        - Phát hiện thư rác tự động
        - Lưu vào database
        - Hiển thị kết quả
        """)
        
        st.subheader("⚠️ Dấu hiệu thư rác")
        st.warning("""
        - Từ khóa về tiền bạc
        - Yêu cầu gấp rút
        - Quà tặng miễn phí
        - Đường link đáng ngờ
        - VIẾT HOA quá nhiều
        - Nhiều dấu chấm than!!!
        """)
        
        st.subheader("🧪 Ví dụ Email Spam")
        if st.button("Xem ví dụ"):
            st.code("""
CONGRATULATIONS!!! 
You have WON $1,000,000!

Click here NOW to claim your prize!
This offer expires TODAY!

Act immediately!!!
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Database
with tab2:
    st.subheader("📊 Email Database")
    
    if len(st.session_state.emails_df) > 0:
        # Bộ lọc
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Lọc theo loại:",
                ["Tất cả", "SPAM", "HAM"]
            )
        
        with col2:
            search_term = st.text_input("🔍 Tìm kiếm:", placeholder="Nhập từ khóa...")
        
        with col3:
            st.metric("Tổng số email", len(st.session_state.emails_df))
        
        # Áp dụng bộ lọc
        filtered_df = st.session_state.emails_df.copy()
        
        if filter_type != "Tất cả":
            filtered_df = filtered_df[filtered_df['Prediction'] == filter_type]
        
        if search_term:
            mask = (
                filtered_df['Subject'].str.contains(search_term, case=False, na=False) |
                filtered_df['Content'].str.contains(search_term, case=False, na=False) |
                filtered_df['From'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        
        # Hiển thị bảng với chiều cao cố định
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Nút tải xuống
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải CSV",
                data=csv,
                file_name=f"emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("🗑️ Xóa toàn bộ"):
                if st.session_state.get('confirm_delete', False):
                    st.session_state.emails_df = pd.DataFrame(
                        columns=['Timestamp', 'From', 'To', 'Subject', 'Content', 'Prediction']
                    )
                    if os.path.exists('emails_database.xlsx'):
                        os.remove('emails_database.xlsx')
                    st.session_state.confirm_delete = False
                    st.rerun()
                else:
                    st.session_state.confirm_delete = True
                    st.warning("⚠️ Nhấn lần nữa để xác nhận xóa!")
    else:
        st.info("📭 Chưa có email nào trong database. Hãy gửi email đầu tiên!")

# Tab 3: Thống kê
with tab3:
    st.subheader("📈 Thống Kê & Phân Tích")
    
    if len(st.session_state.emails_df) > 0:
        col1, col2, col3 = st.columns(3)
        
        # Đếm số lượng
        spam_count = len(st.session_state.emails_df[st.session_state.emails_df['Prediction'] == 'SPAM'])
        ham_count = len(st.session_state.emails_df[st.session_state.emails_df['Prediction'] == 'HAM'])
        total_count = len(st.session_state.emails_df)
        
        with col1:
            st.metric(
                "🚨 Thư rác (SPAM)",
                spam_count,
                f"{(spam_count/total_count*100):.1f}%" if total_count > 0 else "0%"
            )
        
        with col2:
            st.metric(
                "✅ Thư thường (HAM)",
                ham_count,
                f"{(ham_count/total_count*100):.1f}%" if total_count > 0 else "0%"
            )
        
        with col3:
            st.metric(
                "📧 Tổng số email",
                total_count
            )
        
        # Biểu đồ
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
            if 'Timestamp' in st.session_state.emails_df.columns:
                try:
                    time_df = st.session_state.emails_df.copy()
                    time_df['Date'] = pd.to_datetime(time_df['Timestamp']).dt.date
                    daily_count = time_df.groupby('Date').size().reset_index(name='Count')
                    st.line_chart(daily_count.set_index('Date'))
                except:
                    st.info("Không đủ dữ liệu để hiển thị biểu đồ")
        
        # Thêm thông tin chi tiết
        st.subheader("📊 Chi tiết phân tích")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 người gửi email nhiều nhất:**")
            if 'From' in st.session_state.emails_df.columns:
                top_senders = st.session_state.emails_df['From'].value_counts().head(5)
                st.dataframe(top_senders)
        
        with col2:
            st.write("**Top 5 người nhận email nhiều nhất:**")
            if 'To' in st.session_state.emails_df.columns:
                top_recipients = st.session_state.emails_df['To'].value_counts().head(5)
                st.dataframe(top_recipients)
        
    else:
        st.info("📊 Chưa có dữ liệu để hiển thị thống kê.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🔒 Email Spam Detection System | {'Machine Learning' if model else 'Rule-based'} Algorithm</p>
    <p>Hệ thống tự động phát hiện và lọc thư rác</p>
    <p style='font-size: 12px; color: #999;'>💾 Bộ nhớ được tối ưu hóa | ⚡ Xử lý nhanh</p>
</div>
""", unsafe_allow_html=True)