import os
import json
import sys
import importlib.util
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
from sklearn.metrics import accuracy_score, classification_report

from auto_lf.router import Router
from auto_lf.extractors.vlm_wrapper import VLMExtractor
from auto_lf.miners.rule_miner import RuleMiner

# ==========================================
# CẤU HÌNH
# ==========================================
# Đường dẫn dữ liệu Dev (để sinh luật)
DEV_JSON_PATH = "data/devset/dev.json"
# Đường dẫn file sinh ra
OUTPUT_LF_FILE = "generated_lfs.py"

# [CẤU HÌNH TEST]
# Đường dẫn file CSV Test (chứa cột 'image_na' và 'label')
TEST_CSV_PATH = "data/devset/Cifar10-test.csv" 
# Thư mục chứa ảnh Test (vì CSV chỉ có tên file như '0.jpg')
TEST_IMAGES_DIR = "data/devset" 

ABSTAIN = -1

# ==========================================
# HÀM PHỤ TRỢ
# ==========================================
def load_generated_lfs_module(path):
    """Load động file python vừa sinh ra để dùng ngay lập tức"""
    if not os.path.exists(path):
        return []
    
    spec = importlib.util.spec_from_file_location("generated_lfs_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_lfs_module"] = module
    spec.loader.exec_module(module)
    
    lfs = []
    for name in dir(module):
        obj = getattr(module, name)
        # Lấy tất cả hàm bắt đầu bằng auto_lf_
        if callable(obj) and name.startswith("auto_lf"):
            lfs.append(obj)
    return lfs

def predict_with_lfs(lfs, features):
    """
    Áp dụng danh sách LFs lên một tập đặc trưng của 1 ảnh.
    Trả về: Nhãn dự đoán (Majority Vote).
    """
    if not lfs: return ABSTAIN
    
    # Tạo object giả lập có thuộc tính .features để khớp với code trong generated_lfs.py
    x_proxy = SimpleNamespace(features=set(features))
    
    votes = []
    for lf in lfs:
        try:
            vote = lf(x_proxy)
            if vote != ABSTAIN:
                votes.append(vote)
        except:
            pass
            
    if not votes:
        return ABSTAIN
    
    # Lấy nhãn xuất hiện nhiều nhất (Majority Voting)
    return max(set(votes), key=votes.count)

# ==========================================
# MAIN PROCESS
# ==========================================
def main():
    # ---------------------------------------------------------
    # PHẦN 1: SINH LUẬT (MINING PHASE)
    # ---------------------------------------------------------
    if not os.path.exists(DEV_JSON_PATH):
        print(f"❌ LỖI: Không tìm thấy file {DEV_JSON_PATH}")
        return

    print(f"--- 1. LOADING DEV DATA TỪ {DEV_JSON_PATH} ---")
    with open(DEV_JSON_PATH, "r") as f:
        dev_data = json.load(f)

    # Khởi tạo Modules (Dùng chung cho cả Dev và Test)
    router = Router()
    vlm = VLMExtractor(device='cpu') # Sửa thành 'cuda' nếu có GPU
    
    all_features = []
    clean_labels = []

    print(f"--- 2. TRÍCH XUẤT ĐẶC TRƯNG CHO DEV SET ({len(dev_data)} ảnh) ---")
    for item in tqdm(dev_data, desc="Mining Features"):
        img_path = item["path"]
        label = item["label"]
        
        if not os.path.exists(img_path):
             # Fix path tương đối nếu cần
             if os.path.exists(os.path.join(".", img_path)): img_path = os.path.join(".", img_path)
             else: continue

        try:
            mode = router.route(img_path)
            # Logic extract
            feats = vlm.extract(img_path)
            
            all_features.append(feats)
            clean_labels.append(label)
        except Exception as e:
            continue

    if len(all_features) == 0:
        print("❌ FAILED: Không trích xuất được đặc trưng nào.")
        return

    print(f"\n--- 3. MINING RULES & SINH CODE ---")
    miner = RuleMiner()
    miner.fit_and_generate(all_features, clean_labels, output_file=OUTPUT_LF_FILE)
    print(f"✅ Code LF đã được sinh ra tại: '{OUTPUT_LF_FILE}'")

    # ---------------------------------------------------------
    # PHẦN 2: ĐÁNH GIÁ (EVALUATION PHASE)
    # ---------------------------------------------------------
    if not os.path.exists(TEST_CSV_PATH):
        print(f"\n⚠️ Không tìm thấy file {TEST_CSV_PATH} để đánh giá. Kết thúc.")
        return

    print(f"\n--- 4. ĐÁNH GIÁ ĐỘ CHÍNH XÁC TRÊN TEST SET ---")
    
    # A. Load LFs
    lfs = load_generated_lfs_module(OUTPUT_LF_FILE)
    print(f"   -> Đã load được {len(lfs)} hàm LFs.")
    
    if len(lfs) == 0:
        print("❌ Không tìm thấy LF nào để chạy test.")
        return

    # B. Đọc CSV Test
    try:
        df_test = pd.read_csv(TEST_CSV_PATH)
        if 'image_name' not in df_test.columns or 'label' not in df_test.columns:
            print("❌ CSV Test phải có cột 'image_name' và 'label'")
            print(f"   (Các cột hiện có: {list(df_test.columns)})")
            return
    except Exception as e:
        print(f"❌ Lỗi đọc CSV: {e}")
        return

    y_true = []
    y_pred = []
    
    # C. Loop qua tập test để dự đoán
    print(f"   -> Đang chạy LFs trên {len(df_test)} ảnh test...")
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Testing"):
        # Ghép thư mục ảnh với tên file trong CSV
        filename = row['image_na']
        true_label = int(row['label'])
        
        img_full_path = os.path.join(TEST_IMAGES_DIR, filename)
        
        if not os.path.exists(img_full_path):
            # Thử tìm ở thư mục hiện tại nếu path trong csv đã đầy đủ
            if os.path.exists(filename): img_full_path = filename
            else: continue # Bỏ qua nếu không thấy ảnh
            
        try:
            # 1. Trích xuất đặc trưng cho ảnh Test (Bắt buộc để chạy LF)
            test_feats = vlm.extract(img_full_path)
            
            # 2. Dự đoán bằng các LFs vừa sinh ra
            pred_label = predict_with_lfs(lfs, test_feats)
            
            # Chỉ tính các trường hợp LF đưa ra dự đoán (không tính ABSTAIN)
            # Hoặc tùy bạn muốn tính ABSTAIN là sai hay bỏ qua. 
            # Ở đây tôi sẽ tính ABSTAIN (-1) là sai nếu nhãn thật != -1
            y_true.append(true_label)
            y_pred.append(pred_label)
            
        except Exception as e:
            print(f"Lỗi ảnh {filename}: {e}")
            continue

    # D. Tính toán Metrics
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        print("\n" + "="*30)
        print(f"📊 KẾT QUẢ ĐÁNH GIÁ")
        print("="*30)
        print(f"Total Images: {len(y_true)}")
        print(f"ACCURACY    : {acc:.4f} ({acc*100:.2f}%)")
        print("-" * 30)
        
        # In báo cáo chi tiết (Precision/Recall từng class)
        # Filter các nhãn -1 (Abstain) để report đẹp hơn nếu muốn
        print("\nDetailed Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        # Lưu kết quả dự đoán ra file mới nếu cần
        df_test['predicted_label'] = pd.Series(y_pred) # Lưu ý độ dài có thể lệch nếu skip ảnh
        df_test.to_csv("test_predictions.csv", index=False)
        print("Results saved to 'test_predictions.csv'")
    else:
        print("\n❌ Không có dữ liệu test hợp lệ.")

if __name__ == "__main__":
    main()