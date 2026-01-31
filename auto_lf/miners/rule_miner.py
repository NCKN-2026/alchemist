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
            f.write("from wrench.labeling import labeling_function\n")
            f.write("ABSTAIN = -1\n\n")

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
                # class_weight='balanced' giúp cây chú ý đến class hiếm
                clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
                clf.fit(X, binary_labels)
                
                # --- DUYỆT CÂY (Chỉ lấy nhánh dự đoán ra 1 - tức là ra target_class) ---
                tree_ = clf.tree_
                
                def recurse(node, conditions):
                    # Nếu là lá
                    if tree_.feature[node] == -2:
                        # Kiểm tra xem lá này dự đoán là 1 (Target Class) hay 0 (Others)
                        # tree_.value[node] trả về [[số lượng 0, số lượng 1]]
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
                            f.write(f"        return {target_class}\n") # Trả về đúng nhãn target
                            f.write(f"    return ABSTAIN\n\n")
                        return

                    # Nếu là nút nhánh
                    feature_name = feature_names[tree_.feature[node]]
                    recurse(tree_.children_left[node], conditions + [(feature_name, False)])
                    recurse(tree_.children_right[node], conditions + [(feature_name, True)])

                # Bắt đầu duyệt cây của class hiện tại
                recurse(0, [])