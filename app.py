# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import sqlite3
# from datetime import datetime
# from pathlib import Path
# import os
# import sys

# # Thử import joblib, nếu không có thì dùng pickle
# try:
#     import joblib
#     USE_JOBLIB = True
# except ImportError:
#     import pickle
#     USE_JOBLIB = False
#     st.warning("⚠️ Joblib không có sẵn, đang dùng pickle. Cài đặt: pip install joblib")

# # Cấu hình trang
# st.set_page_config(
#     page_title="Email Spam Detection System",
#     page_icon="📧",
#     layout="wide"
# )

# # CSS
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         padding: 20px;
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#         color: white;
#         margin-bottom: 30px;
#     }
#     .email-form {
#         background-color: #f0f2f6;
#         padding: 30px;
#         border-radius: 15px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     .result-box {
#         padding: 20px;
#         border-radius: 10px;
#         margin-top: 20px;
#         font-size: 18px;
#         font-weight: bold;
#         text-align: center;
#     }
#     .spam {
#         background-color: #ffebee;
#         color: #c62828;
#         border: 2px solid #c62828;
#     }
#     .ham {
#         background-color: #e8f5e9;
#         color: #2e7d32;
#         border: 2px solid #2e7d32;
#     }
#     .stButton>button {
#         width: 100%;
#         background-color: #667eea;
#         color: white;
#         font-size: 18px;
#         font-weight: bold;
#         padding: 15px;
#         border-radius: 10px;
#         border: none;
#         transition: all 0.3s;
#     }
#     .stButton>button:hover {
#         background-color: #764ba2;
#         transform: translateY(-2px);
#     }
#     .model-info {
#         background-color: #e3f2fd;
#         padding: 15px;
#         border-radius: 10px;
#         border-left: 4px solid #2196f3;
#         margin: 10px 0;
#     }
#     .warning-box {
#         background-color: #fff3cd;
#         padding: 15px;
#         border-radius: 10px;
#         border-left: 4px solid #ffc107;
#         margin: 10px 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Khởi tạo session state
# if 'emails_df' not in st.session_state:
#     csv_file = 'emails_database.csv'
#     if os.path.exists(csv_file):
#         try:
#             st.session_state.emails_df = pd.read_csv(csv_file)
#         except Exception as e:
#             st.session_state.emails_df = pd.DataFrame(
#                 columns=['Timestamp', 'From', 'To', 'Subject', 'Content', 'Prediction', 'Confidence']
#             )
#     else:
#         st.session_state.emails_df = pd.DataFrame(
#             columns=['Timestamp', 'From', 'To', 'Subject', 'Content', 'Prediction', 'Confidence']
#         )

# # Load model với joblib
# @st.cache_resource
# def load_trained_model():
#     """
#     Load model Decision Tree với joblib - hỗ trợ cross-version
#     """
#     model_path = 'model_rf.mdl'
    
#     # Kiểm tra file tồn tại
#     if not os.path.exists(model_path):
#         return None, None, f"❌ File '{model_path}' không tồn tại trong thư mục: {os.getcwd()}"
    
#     # Kiểm tra kích thước file
#     file_size = os.path.getsize(model_path)
#     if file_size == 0:
#         return None, None, f"❌ File '{model_path}' rỗng (0 bytes)"
    
#     try:
#         # Phương pháp 1: Load với joblib (ưu tiên)
#         if USE_JOBLIB:
#             try:
#                 with st.spinner(f'⏳ Đang load model với joblib... ({file_size/1024:.1f} KB)'):
#                     model = joblib.load(model_path)
#                 load_method = "joblib"
#             except Exception as joblib_error:
#                 # Nếu joblib fail, thử pickle với encoding latin1 (cross-version)
#                 st.warning(f"⚠️ Joblib error: {str(joblib_error)[:100]}")
#                 st.info("🔄 Đang thử load với pickle (cross-version mode)...")
                
#                 import pickle
#                 with open(model_path, 'rb') as f:
#                     model = pickle.load(f, encoding='latin1')
#                 load_method = "pickle (latin1 encoding)"
#         else:
#             # Phương pháp 2: Load với pickle + latin1 encoding (hỗ trợ Python 2->3)
#             import pickle
#             with st.spinner(f'⏳ Đang load model với pickle... ({file_size/1024:.1f} KB)'):
#                 with open(model_path, 'rb') as f:
#                     model = pickle.load(f, encoding='latin1')
#             load_method = "pickle (latin1 encoding)"
        
#         # Lấy feature names
#         if hasattr(model, 'feature_names_in_'):
#             feature_names = model.feature_names_in_.tolist()
#         elif hasattr(model, 'feature_names_'):
#             feature_names = model.feature_names_.tolist()
#         elif hasattr(model, 'n_features_in_'):
#             # Tạo feature names mặc định dựa vào số features
#             n_features = model.n_features_in_
#             feature_names = [f"feature_{i}" for i in range(n_features)]
#             st.info(f"ℹ️ Tạo {n_features} feature names mặc định")
#         else:
#             # Last resort: thử predict test để biết số features
#             try:
#                 test_input = np.zeros((1, 3000))  # Thử với 3000 features
#                 model.predict(test_input)
#                 feature_names = [f"feature_{i}" for i in range(3000)]
#                 st.info("ℹ️ Tạo 3000 feature names mặc định")
#             except:
#                 feature_names = None
        
#         # Verify model hoạt động
#         if feature_names:
#             try:
#                 test_vector = np.zeros((1, len(feature_names)))
#                 _ = model.predict(test_vector)
#                 st.success("✅ Model test thành công!")
#             except Exception as test_error:
#                 st.warning(f"⚠️ Model test warning: {str(test_error)[:100]}")
        
#         return model, feature_names, f"✅ Model loaded với {load_method}! (Size: {file_size/1024:.1f} KB)"
        
#     except Exception as e:
#         error_msg = str(e)
        
#         # Xử lý lỗi cụ thể với hướng dẫn chi tiết
#         if "STACK_GLOBAL" in error_msg or "unsupported pickle protocol" in error_msg:
#             fix_instructions = f"""
# ❌ Lỗi Python version không tương thích!

# **Vấn đề:** Model được train bằng Python cũ hơn, không load được trên Python 3.13

# **Giải pháp nhanh - Train lại model:**

# 1. Mở Jupyter Notebook nơi bạn train model
# 2. Chạy lại đoạn code train model
# 3. Lưu lại với code này:

# ```python
# import joblib
# import sys

# print(f"Python version: {{sys.version}}")
# print(f"Đang lưu model...")

# # Lưu model với protocol 4 (tương thích Python 3.4+)
# joblib.dump(model, 'model_tree.pkl', protocol=4)

# print("✅ Đã lưu model_tree.pkl")
# ```

# 4. Copy file model_tree.pkl mới vào thư mục: {os.getcwd()}
# 5. Refresh lại Streamlit app

# **Hoặc dùng Rule-based algorithm tạm thời (độ chính xác ~75%)**
# """
#             return None, None, fix_instructions
        
#         elif "ModuleNotFoundError" in error_msg or "No module named" in error_msg:
#             missing_module = error_msg.split("'")[1] if "'" in error_msg else "unknown"
#             return None, None, f"""
# ❌ Thiếu thư viện: {missing_module}

# **Cài đặt:**
# ```
# pip install {missing_module}
# ```

# Hoặc cài tất cả:
# ```
# pip install joblib scikit-learn numpy pandas
# ```
# """
        
#         elif "sklearn" in error_msg.lower():
#             return None, None, f"""
# ❌ Lỗi scikit-learn version!

# **Giải pháp:**
# ```
# pip install --upgrade scikit-learn joblib
# ```

# Sau đó train lại model hoặc refresh app.

# Chi tiết lỗi: {error_msg[:200]}
# """
        
#         else:
#             return None, None, f"""
# ❌ Lỗi không xác định: {error_msg[:300]}

# **Thử các bước sau:**

# 1. Cài đặt đầy đủ thư viện:
# ```
# pip install --upgrade joblib scikit-learn numpy pandas
# ```

# 2. Train lại model trong Jupyter Notebook:
# ```python
# import joblib
# joblib.dump(model, 'model_tree.pkl', protocol=4)
# ```

# 3. Kiểm tra file không bị corrupt:
# - File size: {file_size} bytes
# - Location: {os.path.abspath(model_path)}

# 4. Hoặc dùng Rule-based algorithm tạm thời
# """


# def extract_features_from_email(email_text, feature_names):
#     """
#     Trích xuất features từ email
#     """
#     if feature_names is None:
#         return None
    
#     try:
#         # Xử lý văn bản
#         clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', email_text)
#         words = clean_text.lower().split()
        
#         # Tạo vector frequency
#         email_vector = np.zeros(len(feature_names), dtype=np.int32)
        
#         for i, feature in enumerate(feature_names):
#             count = words.count(feature.lower())
#             email_vector[i] = count
        
#         return email_vector
#     except Exception as e:
#         st.error(f"Lỗi extract features: {str(e)}")
#         return None

# def predict_with_trained_model(email_text, model, feature_names):
#     """
#     Dự đoán với model đã train
#     """
#     try:
#         features = extract_features_from_email(email_text, feature_names)
        
#         if features is None:
#             return None, None
        
#         # Reshape để predict
#         features = features.reshape(1, -1)
        
#         # Predict
#         prediction = model.predict(features)[0]
        
#         # Lấy confidence
#         try:
#             if hasattr(model, 'predict_proba'):
#                 proba = model.predict_proba(features)[0]
#                 confidence = proba[prediction] * 100
#             else:
#                 confidence = None
#         except:
#             confidence = None
        
#         return int(prediction), confidence
        
#     except Exception as e:
#         st.error(f"Lỗi dự đoán: {str(e)}")
#         return None, None

# def predict_email_fallback(email_text):
#     """
#     Thuật toán dự phòng - Rule-based
#     """
#     spam_keywords = [
#         # Money & Prizes
#         'free', 'win', 'winner', 'prize', 'cash', 'money', 'dollars', 
#         'million', 'billion', 'reward', 'bonus', 'gift', 'earn',
        
#         # Urgency
#         'urgent', 'immediately', 'act now', 'hurry', 'limited time',
#         'expire', 'deadline', 'last chance', 'today only', 'fast',
        
#         # Deception
#         'congratulations', 'selected', 'claim', 'verify', 'confirm',
#         'account', 'password', 'update', 'suspended', 'locked',
        
#         # Marketing
#         'click here', 'click now', 'subscribe', 'unsubscribe',
#         'offer', 'discount', 'cheap', 'lowest price', 'deal', 'sale',
        
#         # Medical
#         'viagra', 'cialis', 'pills', 'pharmacy', 'medication',
#         'weight loss', 'lose weight', 'drug',
        
#         # Finance
#         'credit card', 'loan', 'debt', 'investment', 'bitcoin',
#         'forex', 'trading', 'casino', 'lottery', 'profit'
#     ]
    
#     text_lower = email_text.lower()
#     spam_score = 0
#     found_keywords = []
    
#     # Đếm keywords
#     for keyword in spam_keywords:
#         if keyword in text_lower:
#             spam_score += 1
#             found_keywords.append(keyword)
    
#     # Kiểm tra các dấu hiệu khác
#     if text_lower.count('!') > 2:
#         spam_score += 1
#     if text_lower.count('$') > 1:
#         spam_score += 1
#     if sum(1 for c in email_text if c.isupper()) > len(email_text) * 0.3:
#         spam_score += 1
#     if len(re.findall(r'http[s]?://', text_lower)) > 2:
#         spam_score += 1
    
#     # Quyết định
#     is_spam = spam_score >= 2
#     confidence = min(spam_score * 12 + 40, 95) if is_spam else max(100 - spam_score * 12, 60)
    
#     return 1 if is_spam else 0, confidence, found_keywords

# # Load model
# with st.spinner('⏳ Đang load model...'):
#     model, feature_names, load_message = load_trained_model()

# # Header
# st.markdown('<div class="main-header"><h1>📧 Hệ Thống Phát Hiện Email Rác</h1><p>Mail Server Simulation & Spam Detection</p></div>', unsafe_allow_html=True)

# # Hiển thị trạng thái model
# if model is not None:
#     st.markdown(f"""
#     <div class="model-info">
#         <h4>✅ Model hoạt động tốt!</h4>
#         <p><strong>Model:</strong> Decision Tree Classifier</p>
#         <p><strong>Load method:</strong> {load_message.split('với')[1].split('!')[0].strip() if 'với' in load_message else 'joblib/pickle'}</p>
#         <p><strong>Features:</strong> {len(feature_names) if feature_names else 'N/A'} từ khóa</p>
#         <p><strong>Accuracy:</strong> ~91.69% (từ training)</p>
#         <p><strong>Python:</strong> {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}</p>
#         <p><strong>Scikit-learn:</strong> {__import__('sklearn').__version__ if 'sklearn' in sys.modules else 'Not imported yet'}</p>
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     st.markdown(f"""
#     <div class="warning-box">
#         <h4>⚠️ Đang dùng Rule-based Algorithm</h4>
#         <p><strong>Lý do:</strong> {load_message}</p>
#         <p><strong>Giải pháp:</strong></p>
#         <ul>
#             <li>Đảm bảo file <code>model_tree.pkl</code> nằm cùng thư mục với app.py</li>
#             <li>Cài đặt đầy đủ thư viện: <code>pip install joblib scikit-learn pandas numpy</code></li>
#             <li>Train lại model với Python {sys.version_info.major}.{sys.version_info.minor}</li>
#             <li>Hoặc dùng thuật toán dự phòng (độ chính xác ~75%)</li>
#         </ul>
#         <p><strong>Thư mục hiện tại:</strong> <code>{os.getcwd()}</code></p>
#         <p><strong>Files .pkl có sẵn:</strong> {', '.join([f for f in os.listdir('.') if f.endswith('.pkl')]) or 'Không có'}</p>
#         <p><strong>Joblib:</strong> {'✅ Có sẵn' if USE_JOBLIB else '❌ Chưa cài (pip install joblib)'}</p>
#     </div>
#     """, unsafe_allow_html=True)

# # Tạo tabs
# tab1, tab2, tab3, tab4 = st.tabs(["📨 Gửi Email", "📊 Database", "📈 Thống Kê", "🧪 Test"])

# # Tab 1: Gửi Email
# with tab1:
#     st.markdown('<div class="email-form">', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.subheader("✉️ Soạn Email Mới")
        
#         with st.form("email_form"):
#             from_email = st.text_input(
#                 "Từ (From):",
#                 value="user@example.com",
#                 help="Địa chỉ email người gửi"
#             )
            
#             to_email = st.text_input(
#                 "Đến (To):",
#                 value="recipient@example.com",
#                 help="Địa chỉ email người nhận"
#             )
            
#             subject = st.text_input(
#                 "Tiêu đề (Subject):",
#                 placeholder="Nhập tiêu đề email...",
#                 help="Tiêu đề của email"
#             )
            
#             content = st.text_area(
#                 "Nội dung (Content):",
#                 placeholder="Nhập nội dung email...",
#                 height=200,
#                 help="Nội dung chi tiết của email"
#             )
            
#             submit_button = st.form_submit_button("📤 Gửi Email", use_container_width=True)
        
#         if submit_button:
#             if not from_email or not to_email or not subject or not content:
#                 st.error("❌ Vui lòng điền đầy đủ thông tin!")
#             else:
#                 full_text = f"{subject} {content}"
                
#                 # Dự đoán
#                 prediction = None
#                 confidence = None
#                 found_keywords = []
                
#                 if model is not None and feature_names is not None:
#                     # Dùng ML model
#                     prediction, confidence = predict_with_trained_model(full_text, model, feature_names)
#                     prediction_method = "🤖 Machine Learning Model"
#                 else:
#                     # Dùng rule-based
#                     prediction, confidence, found_keywords = predict_email_fallback(full_text)
#                     prediction_method = "📏 Rule-based Algorithm (Fallback)"
                
#                 if prediction is not None:
#                     # Lưu vào database
#                     new_email = {
#                         'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         'From': from_email,
#                         'To': to_email,
#                         'Subject': subject,
#                         'Content': content,
#                         'Prediction': 'SPAM' if prediction == 1 else 'HAM',
#                         'Confidence': f"{confidence:.1f}%" if confidence else "N/A"
#                     }
                    
#                     new_row = pd.DataFrame([new_email])
#                     st.session_state.emails_df = pd.concat([
#                         st.session_state.emails_df,
#                         new_row
#                     ], ignore_index=True)
                    
#                     # Lưu file
#                     try:
#                         st.session_state.emails_df.to_csv('emails_database.csv', index=False)
#                     except Exception as e:
#                         st.warning(f"⚠️ Không thể lưu file: {str(e)}")
                    
#                     # Hiển thị kết quả
#                     st.success("✅ Email đã được gửi và phân tích!")
#                     st.info(f"**Phương pháp:** {prediction_method}")
                    
#                     if prediction == 1:
#                         st.markdown(
#                             '<div class="result-box spam">🚨 CẢNH BÁO: Email này là THƯ RÁC (SPAM)</div>',
#                             unsafe_allow_html=True
#                         )
#                         if confidence:
#                             st.error(f"**Độ tin cậy:** {confidence:.1f}%")
#                         if found_keywords:
#                             st.warning(f"**Keywords phát hiện:** {', '.join(found_keywords[:10])}")
#                     else:
#                         st.markdown(
#                             '<div class="result-box ham">✅ Email này là THƯ THÔNG THƯỜNG (HAM)</div>',
#                             unsafe_allow_html=True
#                         )
#                         if confidence:
#                             st.success(f"**Độ tin cậy:** {confidence:.1f}%")
#                 else:
#                     st.error("❌ Không thể phân tích email!")
    
#     with col2:
#         st.subheader("📖 Hướng dẫn")
#         st.info("""
#         **Cách sử dụng:**
        
#         1. ✏️ Điền thông tin email
#         2. 📤 Nhấn "Gửi Email"
#         3. 🤖 AI phân tích tự động
#         4. 👀 Xem kết quả chi tiết
#         5. 💾 Dữ liệu được lưu
        
#         **Độ chính xác:**
#         - ML Model: ~91.69%
#         - Rule-based: ~75%
#         """)
        
#         st.subheader("🔧 Debug Info")
#         debug_info = f"""
# Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
# Joblib: {'✅ Available' if USE_JOBLIB else '❌ Not installed'}
# Model: {'Loaded ✅' if model else 'Not loaded ❌'}
# Features: {len(feature_names) if feature_names else 'N/A'}
# Working dir: {os.getcwd()}
# Model file: {'✅ Exists' if os.path.exists('model_tree.pkl') else '❌ Not found'}
#         """
        
#         # Thêm thông tin scikit-learn nếu có
#         try:
#             import sklearn
#             debug_info += f"\nScikit-learn: {sklearn.__version__}"
#         except:
#             debug_info += "\nScikit-learn: ❌ Not installed"
        
#         st.code(debug_info)
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # Tab 2: Database
# with tab2:
#     st.subheader("📊 Email Database")
    
#     if len(st.session_state.emails_df) > 0:
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             filter_type = st.selectbox("Lọc theo loại:", ["Tất cả", "SPAM", "HAM"])
        
#         with col2:
#             sort_by = st.selectbox("Sắp xếp:", ["Mới nhất", "Cũ nhất"])
        
#         with col3:
#             search_term = st.text_input("🔍 Tìm kiếm:")
        
#         with col4:
#             st.metric("📧 Tổng số", len(st.session_state.emails_df))
        
#         # Áp dụng filter
#         filtered_df = st.session_state.emails_df.copy()
        
#         if filter_type != "Tất cả":
#             filtered_df = filtered_df[filtered_df['Prediction'] == filter_type]
        
#         if search_term:
#             mask = (
#                 filtered_df['Subject'].str.contains(search_term, case=False, na=False) |
#                 filtered_df['Content'].str.contains(search_term, case=False, na=False) |
#                 filtered_df['From'].str.contains(search_term, case=False, na=False) |
#                 filtered_df['To'].str.contains(search_term, case=False, na=False)
#             )
#             filtered_df = filtered_df[mask]
        
#         if sort_by == "Mới nhất":
#             filtered_df = filtered_df.sort_values('Timestamp', ascending=False)
#         else:
#             filtered_df = filtered_df.sort_values('Timestamp', ascending=True)
        
#         st.dataframe(filtered_df, use_container_width=True, height=400)
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             csv = filtered_df.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label="📥 Tải CSV",
#                 data=csv,
#                 file_name=f"emails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )
        
#         with col2:
#             if st.button("🗑️ Xóa toàn bộ", use_container_width=True):
#                 if st.session_state.get('confirm_delete', False):
#                     st.session_state.emails_df = pd.DataFrame(
#                         columns=['Timestamp', 'From', 'To', 'Subject', 'Content', 'Prediction', 'Confidence']
#                     )
#                     if os.path.exists('emails_database.csv'):
#                         os.remove('emails_database.csv')
#                     st.session_state.confirm_delete = False
#                     st.rerun()
#                 else:
#                     st.session_state.confirm_delete = True
#                     st.warning("⚠️ Nhấn lần nữa để xác nhận!")
#     else:
#         st.info("🔭 Chưa có email nào. Hãy gửi email đầu tiên!")

# # Tab 3: Thống kê
# with tab3:
#     st.subheader("📈 Thống Kê & Phân Tích")
    
#     if len(st.session_state.emails_df) > 0:
#         spam_count = len(st.session_state.emails_df[st.session_state.emails_df['Prediction'] == 'SPAM'])
#         ham_count = len(st.session_state.emails_df[st.session_state.emails_df['Prediction'] == 'HAM'])
#         total_count = len(st.session_state.emails_df)
        
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("🚨 Thư rác", spam_count, f"{(spam_count/total_count*100):.1f}%")
        
#         with col2:
#             st.metric("✅ Thư thường", ham_count, f"{(ham_count/total_count*100):.1f}%")
        
#         with col3:
#             st.metric("📧 Tổng số", total_count)
        
#         with col4:
#             st.metric("🤖 Model", "ML" if model else "Rule")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Phân bổ loại email")
#             chart_data = pd.DataFrame({
#                 'Loại': ['SPAM', 'HAM'],
#                 'Số lượng': [spam_count, ham_count]
#             })
#             st.bar_chart(chart_data.set_index('Loại'))
        
#         with col2:
#             st.subheader("Email theo thời gian")
#             try:
#                 time_df = st.session_state.emails_df.copy()
#                 time_df['Date'] = pd.to_datetime(time_df['Timestamp']).dt.date
#                 daily_count = time_df.groupby('Date').size().reset_index(name='Count')
#                 st.line_chart(daily_count.set_index('Date'))
#             except:
#                 st.info("Chưa đủ dữ liệu")
#     else:
#         st.info("📊 Chưa có dữ liệu để hiển thị thống kê.")

# # Tab 4: Test
# with tab4:
#     st.subheader("🧪 Test Email Spam Detection")
    
#     test_email = st.text_area(
#         "Nhập nội dung email để test:",
#         placeholder="Nhập tiêu đề và nội dung email...",
#         height=150
#     )
    
#     if st.button("🔍 Phân tích ngay", use_container_width=True):
#         if test_email:
#             if model is not None and feature_names is not None:
#                 prediction, confidence = predict_with_trained_model(test_email, model, feature_names)
#                 method = "🤖 ML Model"
#             else:
#                 prediction, confidence, found_keywords = predict_email_fallback(test_email)
#                 method = "📏 Rule-based"
            
#             if prediction is not None:
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     if prediction == 1:
#                         st.error("🚨 SPAM")
#                     else:
#                         st.success("✅ HAM")
                
#                 with col2:
#                     if confidence:
#                         st.metric("Độ tin cậy", f"{confidence:.1f}%")
                
#                 with col3:
#                     st.info(f"**Phương pháp:** {method}")
#             else:
#                 st.error("❌ Không thể phân tích!")
#         else:
#             st.warning("⚠️ Vui lòng nhập nội dung!")
    
#     st.markdown("---")
#     st.markdown("### 📝 Ví dụ email")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("**❌ SPAM:**")
#         spam_ex = """CONGRATULATIONS!!! You WON $1,000,000!
# Click here NOW! Limited time offer!
# Verify account immediately or lose prize!"""
#         st.code(spam_ex)
    
#     with col2:
#         st.markdown("**✅ HAM:**")
#         ham_ex = """Meeting Reminder: Project Review
# Hi team, reminder about tomorrow's meeting at 2pm.
# Please bring your progress reports."""
#         st.code(ham_ex)

# # Footer
# st.markdown("---")
# st.markdown(f"""
# <div style='text-align: center; color: #666; padding: 20px;'>
#     <p>📧 Email Spam Detection System | {'ML Model (91.69%)' if model else 'Rule-based (~75%)'}</p>
#     <p>🐍 Python {sys.version_info.major}.{sys.version_info.minor} | 💾 Features: {len(feature_names) if feature_names else 'N/A'}</p>
# </div>
# """, unsafe_allow_html=True)
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

# CSS
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
    }
</style>
""", unsafe_allow_html=True)

# Database functions
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
    except:
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

# Khởi tạo database connection
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_database()

# Load model
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
            confidence = proba[prediction] * 100
        
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
    
    st.info("""
    **Yêu cầu format CSV:**
    - Phải có cột: `Subject` (tiêu đề) và `Content` (nội dung)
    - Có thể có thêm: `From`, `To` (nếu không có sẽ để mặc định)
    """)
    
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
            if st.button("🗑️ Xóa toàn bộ", use_container_width=True):
                if st.session_state.get('confirm_delete', False):
                    clear_database(st.session_state.db_conn)
                    st.session_state.confirm_delete = False
                    st.rerun()
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