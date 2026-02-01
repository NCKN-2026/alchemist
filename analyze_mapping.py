import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Import VLM của chúng ta
from auto_lf.extractors.vlm_wrapper import VLMExtractor
from auto_lf.router import Router

# ================= CẤU HÌNH =================
DEV_JSON_PATH = "data/devset/dev.json"  # File chứa ảnh và nhãn dev
# ============================================

def main():
    if not os.path.exists(DEV_JSON_PATH):
        print(f"❌ Không tìm thấy {DEV_JSON_PATH}")
        return

    print(f"--- 1. LOADING DATA ---")
    with open(DEV_JSON_PATH, "r") as f:
        dev_data = json.load(f)

    # Load Model (Chỉ cần CPU là đủ cho việc test nhanh này)
    vlm = VLMExtractor(device='cpu') 
    
    texts = []
    labels = []
    
    print(f"--- 2. EXTRACTING & MAPPING ---")
    # Quét qua dữ liệu để lấy mô tả
    for item in tqdm(dev_data, desc="Analyzing"):
        img_path = item["path"]
        label = item["label"]
        
        if not os.path.exists(img_path):
             if os.path.exists(os.path.join(".", img_path)): img_path = os.path.join(".", img_path)
             else: continue

        try:
            # Lấy features (list các từ)
            feats_list = vlm.extract(img_path)
            # Nối lại thành 1 câu để dễ phân tích thống kê
            text_desc = " ".join(feats_list)
            
            texts.append(text_desc)
            labels.append(label)
        except Exception as e:
            continue

    if len(texts) == 0:
        print("❌ Không trích xuất được dữ liệu nào.")
        return

    # Chuyển sang DataFrame để dễ xử lý
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    print("\n" + "="*40)
    print("📊 PHÂN TÍCH TỪ KHÓA (TOP KEYWORDS)")
    print("="*40)
    
    # In ra top 10 từ khóa đặc trưng cho mỗi class
    unique_labels = sorted(df['label'].unique())
    vectorizer = CountVectorizer(stop_words='english')
    
    for lbl in unique_labels:
        subset = df[df['label'] == lbl]['text']
        if len(subset) == 0: continue
        
        # Đếm từ
        all_words = " ".join(subset).split()
        counter = Counter(all_words)
        top_10 = counter.most_common(10)
        
        print(f"\n🏷️  LABEL {lbl} (Tổng {len(subset)} ảnh):")
        print(f"   Top words: {', '.join([f'{w}({c})' for w, c in top_10])}")

    print("\n" + "="*40)
    print("🧠 DIAGNOSTIC TRAINING (TRAIN THỬ)")
    print("="*40)
    print("Đang train một model Logistic Regression đơn giản trên mô tả...")
    
    # Vector hóa dạng Bag-of-Words
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # Train model
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X, y)
    
    # Đánh giá độ khớp (Accuracy trên chính tập train)
    # Nếu Acc cao -> Mô tả khớp tốt với nhãn
    # Nếu Acc thấp -> BLIP "nhìn gà hóa cuốc" hoặc dữ liệu quá khó
    acc = clf.score(X, y)
    print(f"\n✅ Mapping Accuracy (Training Score): {acc:.4f} ({acc*100:.2f}%)")
    
    if acc < 0.6:
        print("⚠️ CẢNH BÁO: Độ khớp thấp! Có thể mô tả của BLIP không chứa thông tin phân loại.")
    else:
        print("🚀 TỐT: Mô tả văn bản chứa đủ thông tin để phân biệt các nhãn.")

    # In feature importance (Từ nào quan trọng nhất với model)
    if len(unique_labels) == 2: # Chỉ in nếu là bài toán nhị phân cho gọn
        print("\n🔍 TỪ KHÓA QUYẾT ĐỊNH (Feature Importance):")
        feature_names = vectorizer.get_feature_names_out()
        coefs = clf.coef_[0]
        sorted_idx = np.argsort(coefs)
        
        print(f"   Top words cho Label {unique_labels[0]} (Negative coefs):")
        top_neg = sorted_idx[:10]
        print(f"   -> {', '.join([feature_names[i] for i in top_neg])}")
        
        print(f"\n   Top words cho Label {unique_labels[1]} (Positive coefs):")
        top_pos = sorted_idx[-10:]
        print(f"   -> {', '.join([feature_names[i] for i in top_pos])}")

if __name__ == "__main__":
    main()