from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class RuleMiner:
    def fit_and_generate(self, triplets_data, labels, output_file="generated_lfs.py"):
        """
        triplets_data: List[List[str]] - Danh sách các đặc trưng dạng text
        labels: List[int] - Nhãn thật
        """
        # 1. Vector hóa
        # preprocessor=lambda x: x để sk-learn không cố lowercase hoặc tách từ lại
        vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, preprocessor=lambda x: x)
        X = vectorizer.fit_transform(triplets_data)
        feature_names = vectorizer.get_feature_names_out()
        
        # 2. Train cây quyết định
        # max_depth nhỏ (3-4) để luật sinh ra ngắn gọn, dễ hiểu
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X, labels)
        
        # 3. Sinh code Python động dựa trên cấu trúc cây
        print(f"Generating LFs into {output_file}...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            # Header bắt buộc
            f.write("from wrench.labeling import labeling_function\n")
            f.write("ABSTAIN = -1\n\n")
            
            # --- LOGIC DUYỆT CÂY (RECURSIVE) ---
            tree_ = clf.tree_
            
            def recurse(node, conditions):
                # Nếu là nút lá (Leaf node) -> Sinh ra 1 hàm LF
                if tree_.feature[node] == -2: # -2 nghĩa là không có con (lá)
                    # Lấy class dự đoán (class có số mẫu nhiều nhất tại lá này)
                    class_idx = np.argmax(tree_.value[node])
                    predicted_label = clf.classes_[class_idx]
                    
                    # Chỉ sinh luật nếu đường đi này có ý nghĩa (không phải gốc)
                    if len(conditions) > 0:
                        # Tạo tên hàm unique dựa trên ID của node
                        func_name = f"auto_lf_node_{node}"
                        
                        # Viết code hàm
                        f.write(f"@labeling_function()\n")
                        f.write(f"def {func_name}(x):\n")
                        f.write(f"    features = set(x.features)\n")
                        
                        # Ghép các điều kiện thành chuỗi logic
                        # conditions là list các tuple: (tên_feature, phải_có_hay_không)
                        cond_strs = []
                        for feat, present in conditions:
                            if present:
                                cond_strs.append(f"'{feat}' in features")
                            else:
                                cond_strs.append(f"'{feat}' not in features")
                        
                        full_condition = " and ".join(cond_strs)
                        
                        f.write(f"    if {full_condition}:\n")
                        f.write(f"        return {predicted_label}\n")
                        f.write(f"    return ABSTAIN\n\n")
                    return

                # Nếu là nút nhánh (Internal node) -> Tiếp tục đi xuống
                # Lấy tên feature tại nút chia này
                feature_name = feature_names[tree_.feature[node]]
                
                # Cây SKLearn: Left là False (<= 0.5), Right là True (> 0.5) 
                # Vì CountVectorizer trả về 0 hoặc 1, nên <=0.5 nghĩa là KHÔNG CÓ
                
                # Đi sang trái (Điều kiện: KHÔNG CÓ feature này)
                recurse(tree_.children_left[node], conditions + [(feature_name, False)])
                
                # Đi sang phải (Điều kiện: CÓ feature này)
                recurse(tree_.children_right[node], conditions + [(feature_name, True)])

            # Bắt đầu duyệt từ gốc (node 0), chưa có điều kiện gì
            recurse(0, [])