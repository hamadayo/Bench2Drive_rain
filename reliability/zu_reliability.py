import matplotlib.pyplot as plt

# テキストファイルのパスを指定
text_file_path = "/home/yoshi-22/Bench2Drive/eval_v1/attention_1711/RouteScenario_0_rep0_Town12_ParkingCutIn_1_15_01_24_01_39_42/reliability_log_front.txt"

# 値を格納するリスト
reliability_vals = []

with open(text_file_path, "r", encoding="utf-8") as f:
    # 全行を取得
    lines = f.readlines()
    
    # 1〜604行目だけを対象にする
    end_line = min(len(lines), 604)
    for i in range(end_line):
        line = lines[i].strip()
        splitted = line.split(":")
        if len(splitted) == 2:
            val_str = splitted[1].strip()
            try:
                val_float = float(val_str)
                reliability_vals.append(val_float)
            except ValueError:
                continue

# 取得したreliability_valの件数を確認
print(f"Number of reliability values read: {len(reliability_vals)}")

# ----- グラフ描画 -----
plt.figure(figsize=(10, 6))
# 点の表示(マーカー)を消したいので marker の指定を削除するか、marker='' に設定
plt.plot(reliability_vals, linestyle='-')  # marker='o' を削除

plt.title("Reliability Values (Lines 1 to 604)")
plt.xlabel("Index (Line Number)")
plt.ylabel("Reliability Value")

# グリッド表示を消す
plt.grid(False)

plt.show()
