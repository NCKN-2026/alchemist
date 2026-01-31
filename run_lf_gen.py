import os
import json
from tqdm import tqdm # Thêm thư viện này để hiện thanh % tiến trình
from auto_lf.router import Router
from auto_lf.extractors.vlm_wrapper import VLMExtractor
from auto_lf.miners.rule_miner import RuleMiner

def main():
    # 1. Cấu hình đường dẫn (Hardcode hoặc lấy từ args nếu muốn)
    DEV_JSON_PATH = "data/devset/dev.json"
    OUTPUT_FILE = "generated_lfs.py"

    # 2. Kiểm tra file json tồn tại chưa
    if not os.path.exists(DEV_JSON_PATH):
        print(f"❌ LỖI: Không tìm thấy file {DEV_JSON_PATH}")
        print("   -> Hãy tạo file json chứa list ảnh mẫu theo hướng dẫn trước.")
        return

    print(f"Loading dev data from {DEV_JSON_PATH}...")
    with open(DEV_JSON_PATH, "r") as f:
        dev_data = json.load(f)

    # 3. Khởi tạo Modules
    # Lưu ý: Nếu server yếu RAM, hãy giữ device='cpu'. Nếu có GPU mạnh thì sửa thành 'cuda'
    router = Router()
    vlm = VLMExtractor(device='cpu') 
    
    all_features = []
    clean_labels = [] # Lưu label tương ứng với ảnh (đã lọc ảnh lỗi)

    print(f"🚀 Start Extracting features from {len(dev_data)} images...")

    # 4. Loop xử lý ảnh (Dùng tqdm để hiện thanh loading)
    for item in tqdm(dev_data, desc="Processing Images"):
        img_path = item["path"]
        label = item["label"]
        
        # Kiểm tra đường dẫn ảnh có đúng không
        if not os.path.exists(img_path):
            # Thử fix đường dẫn nếu đang đứng ở root (đôi khi json lưu đường dẫn tương đối lạ)
            if os.path.exists(os.path.join(".", img_path)):
                img_path = os.path.join(".", img_path)
            else:
                print(f"\n⚠️ Warning: Image not found at '{img_path}', skipping...")
                continue

        # Logic Router
        try:
            mode = router.route(img_path)
            
            if mode == "SIMPLE":
                feats = vlm.extract(img_path)
            else:
                # Nếu chưa cài PSG, dùng VLM luôn cho ảnh phức tạp (Fallback)
                feats = vlm.extract(img_path) 
            
            all_features.append(feats)
            clean_labels.append(label)
            
        except Exception as e:
            print(f"\n❌ Error extracting {img_path}: {e}")
            continue

    # 5. Sinh luật từ đặc trưng đã rút trích
    if len(all_features) > 0:
        print(f"\n⛏️ Mining Rules from {len(all_features)} valid samples...")
        miner = RuleMiner()
        # Gọi hàm fit_and_generate mà ta đã viết trong class RuleMiner
        miner.fit_and_generate(all_features, clean_labels, output_file=OUTPUT_FILE)
        
        print(f"\n✅ SUCCESS! Code LF đã được sinh ra tại: '{OUTPUT_FILE}'")
        print("   -> Bạn có thể mở file này lên để kiểm tra logic.")
    else:
        print("\n❌ FAILED: Không trích xuất được đặc trưng nào. Hãy kiểm tra lại đường dẫn ảnh.")

if __name__ == "__main__":
    main()