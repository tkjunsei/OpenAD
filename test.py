import pickle

# ファイルのパス
pkl_file_path = 'drive/full_shape_train_data.pkl'

# .pkl ファイルを開く
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# 内容を確認
print(data)
