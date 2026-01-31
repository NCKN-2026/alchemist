from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class RuleMiner:
    def fit_and_generate(self, triplets_data, labels, output_file="generated_lfs.py"):
        """
        Chiến lược One-vs-Rest: Đảm bảo mọi class đều có LF riêng.
        """
        # 1. Vector hóa dữ liệu
        vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, preprocessor=lambda x: x)
        X = vectorizer.fit_transform(triplets_data)
        feature_names = vectorizer.get_feature_names_out()
        
        # Xác định danh sách các nhãn duy nhất (Ví dụ: [0, 1, 2])
        unique_classes = np.unique(labels)
        
        print(f"Generating LFs into {output_file}...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            # === [THAY ĐỔI QUAN TRỌNG Ở ĐÂY] ===
            # XÓA dòng import wrench cũ
            # THAY BẰNG đoạn code tự định nghĩa decorator để không bị lỗi ModuleNotFoundError
            f.write("import logging\n")
            f.write("ABSTAIN = -1\n\n")

            f.write("# Định nghĩa decorator giả lập để chạy độc lập\n")
            f.write("def labeling_function(**kwargs):\n")
            f.write("    def decorator(f):\n")
            f.write("        return f\n")
            f.write("    return decorator\n\n")
            # ===================================

            # --- VÒNG LẶP ONE-VS-REST ---
            for target_class in unique_classes:
                print(f"   > Mining rules for Class {target_class}...")
                
                # Tạo nhãn nhị phân: 1 nếu là target_class, 0 nếu là các class khác
                binary_labels = [1 if y == target_class else 0 for y in labels]
                
                # Nếu class này quá ít mẫu (<2), bỏ qua để tránh lỗi
                if sum(binary_labels) < 2:
                    print(f"     Warning: Class {target_class} has too few samples, skipping.")
                    continue

                # Train cây riêng cho Class này
                clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
                clf.fit(X, binary_labels)
                
                # --- DUYỆT CÂY ---
                tree_ = clf.tree_
                
                def recurse(node, conditions):
                    # Nếu là lá
                    if tree_.feature[node] == -2:
                        prediction = np.argmax(tree_.value[node]) 
                        
                        # CHỈ SINH LUẬT NẾU LÁ NÀY DỰ ĐOÁN LÀ TARGET_CLASS (1)
                        if prediction == 1:
                            func_name = f"auto_lf_class_{target_class}_node_{node}"
                            
                            # Viết hàm
                            f.write(f"@labeling_function()\n")
                            f.write(f"def {func_name}(x):\n")
                            f.write(f"    features = set(x.features)\n")
                            
                            cond_strs = []
                            for feat, present in conditions:
                                if present:
                                    cond_strs.append(f"'{feat}' in features")
                                else:
                                    cond_strs.append(f"'{feat}' not in features")
                            
                            full_condition = " and ".join(cond_strs)
                            
                            f.write(f"    if {full_condition}:\n")
                            f.write(f"        return {target_class}\n") 
                            f.write(f"    return ABSTAIN\n\n")
                        return

                    # Nếu là nút nhánh
                    feature_name = feature_names[tree_.feature[node]]
                    recurse(tree_.children_left[node], conditions + [(feature_name, False)])
                    recurse(tree_.children_right[node], conditions + [(feature_name, True)])

                # Bắt đầu duyệt cây của class hiện tại
                recurse(0, [])