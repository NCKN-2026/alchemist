from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

class RuleMiner:
    def fit_and_generate(self, triplets_data, labels, output_file="generated_lfs.py"):
        """
        triplets_data: List[List[str]] (Mỗi ảnh là 1 list các đặc trưng)
        labels: List[int] (Nhãn thật của ảnh dev)
        """
        # 1. Vector hóa
        # Dùng dummy token để xử lý list of strings
        vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, preprocessor=lambda x: x)
        X = vectorizer.fit_transform(triplets_data)
        feature_names = vectorizer.get_feature_names_out()
        
        # 2. Train cây quyết định
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, labels)
        
        # 3. Sinh code Python
        print(f"Generating LFs into {output_file}...")
        with open(output_file, "w") as f:
            f.write("from wrench.labeling import labeling_function\n")
            f.write("ABSTAIN = -1\n\n")
            
            # Logic duyệt cây để viết code (Simplification)
            # Đây là ví dụ template, bạn cần code phần duyệt cây recursive
            f.write("@labeling_function()\n")
            f.write("def auto_lf_demo(x):\n")
            f.write("    # Lấy đặc trưng từ input x (x cần có thuộc tính .features)\n")
            f.write("    features = set(x.features)\n")
            f.write("    if 'helmet' in features: return 1\n")
            f.write("    return ABSTAIN\n")