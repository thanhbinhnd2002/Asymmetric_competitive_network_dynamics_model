import os
import pandas as pd

# **1. Đọc file kết quả cạnh tranh**
competition_file = "results_test_1/Beta_1/Human Gene Regulatory Network - Input.csv"

# **Lấy tên dataset từ tên file để đặt tên file kết quả**
dataname = os.path.basename(competition_file).replace(".csv", "")

# **Đọc dữ liệu từ file kết quả cạnh tranh**
competition_df = pd.read_csv(competition_file, sep="\t")  # Giả sử file CSV có dấu tab

# **2. Đọc file đối chiếu**
oncogene_file = "Cancer gene OncoKB30012025.xlsx"
oncogene_df = pd.read_excel(oncogene_file)

# **3. Chuẩn hóa dữ liệu trong file đối chiếu**
# Chọn các cột quan trọng và loại bỏ 'hsa:' khỏi mã gen trong file kết quả
oncogene_df = oncogene_df[['Hugo Symbol', 'Entrez Gene ID', 'Is Oncogene', 'Is Tumor Suppressor Gene']]
oncogene_df['Entrez Gene ID'] = oncogene_df['Entrez Gene ID'].astype(str)  # Chuyển mã gen thành chuỗi
oncogene_df['Is Oncogene'] = oncogene_df['Is Oncogene'].astype(str).str.strip().str.capitalize()  # Chuẩn hóa dữ liệu
oncogene_df['Is Tumor Suppressor Gene'] = oncogene_df['Is Tumor Suppressor Gene'].astype(str).str.strip().str.capitalize()

# **4. Xử lý dữ liệu file kết quả cạnh tranh**
# Loại bỏ 'hsa:' khỏi tên gen trong file kết quả
competition_df['Gen A'] = competition_df['Gen A'].str.replace('hsa:', '', regex=True)
competition_df['Gen B'] = competition_df['Gen B'].str.replace('hsa:', '', regex=True)

# **5. Gán loại gen (Oncogene / Tumor Suppressor / Unknown)**
def get_gene_type(gene_id):
    """Xác định loại gen từ file đối chiếu."""
    match = oncogene_df[oncogene_df['Hugo Symbol'] == gene_id]
    if not match.empty:
        types = []
        if match['Is Oncogene'].values[0] == "Yes":  # Kiểm tra giá trị là "Yes"
            types.append("Oncogene")
        if match['Is Tumor Suppressor Gene'].values[0] == "Yes":  # Kiểm tra giá trị là "Yes"
            types.append("Tumor Suppressor")
        return "+".join(types) if types else "Unknown"
    return "Unknown"

# Áp dụng cho từng gene trong danh sách
competition_df['Type A'] = competition_df['Gen A'].apply(get_gene_type)
competition_df['Type B'] = competition_df['Gen B'].apply(get_gene_type)

# **6. Sắp xếp theo Strength từ cao xuống thấp**
competition_df = competition_df.sort_values(by="Strength", ascending=False)

# **7. Tạo thư mục lưu kết quả nếu chưa tồn tại**
output_folder = "final_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# **8. Xuất file kết quả**
output_file = os.path.join(output_folder, f"{dataname}.csv")
competition_df.to_csv(output_file, index=False, sep="\t")

print(f"Kết quả đã được lưu tại: {output_file}")
