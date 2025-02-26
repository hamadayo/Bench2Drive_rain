import os
import re
import math
import matplotlib.pyplot as plt

def parse_file(filepath):
    """
    単一のテキストファイルから Location と Driving score を抽出する。
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

def parse_directory(dir_path):
    """
    指定したディレクトリ内の "live_results_数値.txt" を探し、(all_x, all_y, all_z, all_scores, file_indices) を返す。
    file_indices はファイル名に含まれる数値（例: live_results_10.txt -> 10）。
    """
    pattern = re.compile(r'^live_results_(\d+)\.txt$')
    file_list = []

    for fname in os.listdir(dir_path):
        match = pattern.match(fname)
        if match:
            file_index = int(match.group(1))
            file_list.append((file_index, fname))

    # 数値の小さい順にソート
    file_list.sort(key=lambda x: x[0])

    all_x = []
    all_y = []
    all_z = []
    all_scores = []
    file_indices = []

    # ファイルを順番に読み込む
    for file_index, fname in file_list:
        filepath = os.path.join(dir_path, fname)
        x, y, z, ds = parse_file(filepath)

        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        all_scores.append(ds)
        file_indices.append(file_index)

    return all_x, all_y, all_z, all_scores, file_indices

def compute_l2_distance(x1, y1, x2, y2):
    """
    (x1, y1) と (x2, y2) のL2距離を返す。
    Noneが含まれる場合は None を返す。
    """
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    # 3つのディレクトリ（基準 + 比較対象2つ）
    results_dir  = "/home/yoshi-22/Bench2Drive/eval_v1/gomi_rain_1711/results/3"
    results_dir1 = "/home/yoshi-22/Bench2Drive/eval_v1/gomi_rain_1711/results/6"
    results_dir2 = "/home/yoshi-22/Bench2Drive/eval_v1/gomi_rain_1711/results/8"

    # それぞれ読み込み
    base_x, base_y, base_z, base_scores, base_indices = parse_directory(results_dir)
    x1, y1, z1, scores1, indices1 = parse_directory(results_dir1)
    x2, y2, z2, scores2, indices2 = parse_directory(results_dir2)

    # 軌跡の比較は、「同じファイルインデックス」に対して行う
    # -> base_indices, indices1, indices2 に共通するインデックスのみを扱う
    common_indices = sorted(set(base_indices) & set(indices1) & set(indices2))

    # 軌跡差分を格納するリスト
    diff_list_1 = []  # base vs results_dir1
    diff_list_2 = []  # base vs results_dir2

    # スコア(差が大きいほど小さい)の例: score = 1 / (1 + L2)
    score_list_1 = []
    score_list_2 = []

    # グラフ用に、x軸は common_indices の順番にする
    for idx in common_indices:
        # base
        i_base = base_indices.index(idx)
        bx = base_x[i_base]
        by = base_y[i_base]

        # dir1
        i1 = indices1.index(idx)
        x1v = x1[i1]
        y1v = y1[i1]

        # dir2
        i2 = indices2.index(idx)
        x2v = x2[i2]
        y2v = y2[i2]

        # 軌跡の差分(L2距離)
        dist1 = compute_l2_distance(bx, by, x1v, y1v)
        dist2 = compute_l2_distance(bx, by, x2v, y2v)

        # リストに追加
        diff_list_1.append(dist1)
        diff_list_2.append(dist2)

        # 差が None の場合はスコアも None にしておく
        if dist1 is None:
            score_list_1.append(None)
        else:
            score_list_1.append(1.0 / (1.0 + dist1))

        if dist2 is None:
            score_list_2.append(None)
        else:
            score_list_2.append(1.0 / (1.0 + dist2))

    # --- 図示 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (1) 軌跡の散布図 (3つのディレクトリをまとめて可視化する例)
    #    - 各ディレクトリで持っている (x, y) を全体表示
    #    - None を除去して散布図に
    base_pts  = [(vx, vy) for vx, vy in zip(base_x, base_y) if vx is not None and vy is not None]
    dir1_pts  = [(vx, vy) for vx, vy in zip(x1, y1)       if vx is not None and vy is not None]
    dir2_pts  = [(vx, vy) for vx, vy in zip(x2, y2)       if vx is not None and vy is not None]

    if base_pts:
        bx_vals, by_vals = zip(*base_pts)
        axes[0].scatter(bx_vals, by_vals, c='blue',  marker='o', alpha=0.6, label='Base')
    if dir1_pts:
        x1_vals, y1_vals = zip(*dir1_pts)
        axes[0].scatter(x1_vals, y1_vals, c='red',   marker='^', alpha=0.6, label='Dir1')
    # if dir2_pts:
    #     x2_vals, y2_vals = zip(*dir2_pts)
    #     axes[0].scatter(x2_vals, y2_vals, c='green', marker='s', alpha=0.6, label='Dir2')

    axes[0].set_title("Trajectories Comparison")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    axes[0].legend()
    # x軸の表示範囲を絞りたい場合は例のように: axes[0].set_xlim([-501, -495])

    # (2) 軌跡差分 & スコア を折れ線グラフで可視化
    #    - x軸 = common_indices (ファイルインデックス)
    #    - y軸 = L2距離 or その派生スコア
    
    # Noneを0扱い、あるいはプロット対象外にするかはお好みで
    # ここではシンプルに0にしておきます
    plot_diff_1 = [d if d is not None else 0 for d in diff_list_1]
    plot_diff_2 = [d if d is not None else 0 for d in diff_list_2]
    plot_score_1 = [s if s is not None else 0 for s in score_list_1]
    plot_score_2 = [s if s is not None else 0 for s in score_list_2]

    # 軌跡差分をプロット
    # axes[1].plot(common_indices, plot_diff_1, marker='o', color='red',  label='L2 diff: Base vs Dir1')
    axes[1].plot(common_indices, plot_diff_2, marker='s', color='green',label='Driving scrore')

    # 軌跡差から算出したスコアを、同じグラフに第2軸として重ねたい場合はこんな感じ:
    ax2 = axes[1].twinx()
    ax2.plot(common_indices, plot_score_1, linestyle='--', color='red',  alpha=0.5, label='Score vs Dir1')
    ax2.plot(common_indices, plot_score_2, linestyle='--', color='green',alpha=0.5, label='Score vs Dir2')
    ax2.set_ylabel("Score (1 / (1 + L2 dist))")

    axes[1].set_xlabel("File Index")
    # axes[1].set_ylabel("L2 Distance")
    axes[1].set_title("L2 Distance")
    
    # 凡例をまとめる
    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
