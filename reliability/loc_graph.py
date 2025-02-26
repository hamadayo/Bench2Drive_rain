import os
import re
import matplotlib.pyplot as plt

def parse_file(filepath):
    """
    指定したテキストファイルから Location と Driving score を抽出する。
    戻り値: (x, y, z, driving_score)
    """
    x = y = z = None
    driving_score = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Location行の例: "Location:           (-497.601 3672.927 364.864)"
    location_pattern = re.compile(r'Location:\s*\((-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\)')
    # Driving score行の例: "Driving score: 0.000"
    driving_score_pattern = re.compile(r'Driving score:\s*([\d\.]+)')

    for line in lines:
        loc_match = location_pattern.search(line)
        if loc_match:
            x = float(loc_match.group(1))
            y = float(loc_match.group(2))
            z = float(loc_match.group(3))

        ds_match = driving_score_pattern.search(line)
        if ds_match:
            driving_score = float(ds_match.group(1))

    return x, y, z, driving_score

def main():
    # 読み込み対象のディレクトリ
    # results_dir = "/media/yoshi-22/T7/attention_1711/results/3"
    # 事故 gomi_3,4と０を比較
    # results_dir = "/home/yoshi-22/Bench2Drive/eval_v1/gomi_rain_1711/results/6"
    results_dir = "/home/yoshi-22/Bench2Drive/eval_v1/attention_1711/RouteScenario_0_rep0_Town12_ParkingCutIn_1_15_01_24_01_39_42/0"
    # results_dir1 = "/home/yoshi-22/Bench2Drive/eval_v1/gomi_rain_1711/results/6"
    # results_dir2 = "/home/yoshi-22/Bench2Drive/eval_v1/gomi_rain_1711/results/8"

    # ディレクトリ内の "live_results_数字.txt" をすべて集める
    file_list = []
    pattern = re.compile(r'^live_results_(\d+)\.txt$')  # 例: live_results_0.txt, live_results_10.txt など

    for f in os.listdir(results_dir):
        match = pattern.match(f)
        if match:
            # ファイル名から数字を取り出し、intに変換
            file_index = int(match.group(1))
            file_list.append((file_index, f))

    # 数字 (file_index) でソートして、順番に処理できるようにする
    file_list.sort(key=lambda x: x[0])

    all_x = []
    all_y = []
    all_z = []
    all_driving_scores = []

    # 後でグラフの x 軸に使う用（ファイル名に含まれる数字）のリスト
    file_indices = []

    # ファイルを順番に読み込む
    for file_index, filename in file_list:
        filepath = os.path.join(results_dir, filename)

        x, y, z, ds = parse_file(filepath)
        if x is None or y is None or z is None:
            # 必要に応じて None の場合の対応を入れる
            print(f"Location情報が見つかりません: {filepath}")
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

        if ds is None:
            print(f"Driving scoreが見つかりません: {filepath}")
        all_driving_scores.append(ds)

        file_indices.append(file_index)

    # 図示
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # None を除いたデータだけを散布図にしたい場合はフィルタリング
    # （ここでは単純に None は弾く例を示す）
    plot_x = []
    plot_y = []
    for vx, vy in zip(all_x, all_y):
        if vx is not None and vy is not None:
            plot_x.append(vx)
            plot_y.append(vy)

    # 1. (x, y) の散布図
    axes[0].scatter(plot_x, plot_y, c='blue', marker='o', alpha=0.6)
    axes[0].set_title("Locations (x, y)")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    # 例として x を -507 〜 -487 に固定したい場合
    axes[0].set_xlim([-501, -495])

    # 2. Driving score の推移を折れ線グラフ
    # None を回避するため、None は 0（または他の値）として扱う例
    plot_scores = [ds if ds is not None else 0 for ds in all_driving_scores]

    # x 軸にファイル名に含まれる数値 (file_indices) をそのまま使う
    axes[1].plot(file_indices, plot_scores, marker='o', linestyle='-', color='red')
    axes[1].set_title("Driving scores")
    axes[1].set_xlabel("File Index (extracted from filename)")
    axes[1].set_ylabel("Driving Score")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

