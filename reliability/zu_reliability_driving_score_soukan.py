import os
import re
import numpy as np
import matplotlib.pyplot as plt

def load_reliability_vals(text_file_path, max_lines=604):
    """
    指定テキストファイルから 1〜max_lines 行目までの reliability_val を抽出してリストとして返す。
    行の形式:
        "[CAM_FRONT] reliability_val: 0.2038"
    """
    reliability_vals = []
    with open(text_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        end_line = min(len(lines), max_lines)
        for i in range(end_line):
            line = lines[i].strip()
            splitted = line.split(":")
            if len(splitted) == 2:
                val_str = splitted[1].strip()
                try:
                    val_float = float(val_str)
                    reliability_vals.append(val_float)
                except ValueError:
                    pass  # 数値変換できない場合はスキップ
    return reliability_vals

def parse_file(filepath):
    """
    live_results_*.txt などから location (x, y, z) と driving_score を抽出して返す。
    形式例:
        Location: (-497.601 3672.927 364.864)
        Driving score: 0.000
    """
    x = y = z = None
    driving_score = None

    location_pattern = re.compile(r'Location:\s*\((-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\)')
    driving_score_pattern = re.compile(r'Driving score:\s*([\d\.]+)')

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            loc_match = location_pattern.search(line)
            if loc_match:
                x = float(loc_match.group(1))
                y = float(loc_match.group(2))
                z = float(loc_match.group(3))

            ds_match = driving_score_pattern.search(line)
            if ds_match:
                driving_score = float(ds_match.group(1))

    return x, y, z, driving_score

def load_driving_scores(results_dir):
    """
    指定したディレクトリ内にある live_results_*.txt ファイルを
    ファイル名に含まれる数字順に読み込み、driving_score のリストを返す。
    driving_score が None の場合は 0.0 を代わりに入れる。
    """
    file_list = []
    pattern = re.compile(r'^live_results_(\d+)\.txt$')

    # ディレクトリ内のファイルを列挙
    for f in os.listdir(results_dir):
        match = pattern.match(f)
        if match:
            file_index = int(match.group(1))
            file_list.append((file_index, f))

    # 数値でソート
    file_list.sort(key=lambda x: x[0])

    all_driving_scores = []
    for file_index, filename in file_list:
        filepath = os.path.join(results_dir, filename)
        x, y, z, ds = parse_file(filepath)
        if ds is None:
            ds = 0.0
        all_driving_scores.append(ds)

    return all_driving_scores

def main():
    # --- 1. reliability_val の読み込み ---
    reliability_file = "/home/yoshi-22/Bench2Drive/eval_v1/attention_1711/RouteScenario_0_rep0_Town12_ParkingCutIn_1_15_01_24_01_39_42/reliability_log_front.txt"
    reliability_vals = load_reliability_vals(reliability_file, max_lines=604)

    # --- 2. driving_score の読み込み ---
    results_dir = "/home/yoshi-22/Bench2Drive/eval_v1/attention_1711/RouteScenario_0_rep0_Town12_ParkingCutIn_1_15_01_24_01_39_42/0"
    driving_scores = load_driving_scores(results_dir)

    # --- 3. データの長さをそろえる（小さい方に合わせる） ---
    common_length = min(len(reliability_vals), len(driving_scores))
    rel_sub = reliability_vals[:common_length]
    drv_sub = driving_scores[:common_length]

    # ===============================================================
    # ここから "driving_scoreの微分(差分)" を使って相関を求めるコード例
    # ===============================================================

    # --- 4. driving_score を離散的に微分(差分) ---
    ds_diff = np.diff(drv_sub)  # shape: (common_length - 1,)

    # 例えば「ds_diff[i]」と「rel_sub[i+1]」を対応させる場合は、reliabilityも1個ずらす
    # rel_sub_diff = rel_sub[1:]  # shape: (common_length - 1,)
    #
    # あるいは ds_diff と同じインデックスで対応させたい場合は rel_sub[:-1] を使うなど、
    # 解析目的に応じて決める

    rel_sub_diff = rel_sub[1:]  # 一般的にはこちらを使う例が多い

    # 2つの配列をさらに同じ長さに合わせる
    common_length_diff = min(len(ds_diff), len(rel_sub_diff))
    ds_diff = ds_diff[:common_length_diff]
    rel_sub_diff = rel_sub_diff[:common_length_diff]

    # --- 5. 相関係数(ピアソン)を計算 ---
    if common_length_diff > 1:
        corr_matrix_diff = np.corrcoef(ds_diff, rel_sub_diff)
        corr_coeff_diff = corr_matrix_diff[0, 1]
        print(f"Correlation coefficient (Pearson) between reliability_val and d(driving_score)/dt = {corr_coeff_diff:.4f}")
    else:
        print("有効なデータ点がほとんどないため、微分値との相関係数を計算できません。")

    # --- 6. (任意) グラフ描画 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (A) reliability_val の推移
    axes[0].plot(rel_sub, color='blue')
    axes[0].set_title("Reliability Values")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("reliability_val")

    # (B) driving_score の推移
    axes[1].plot(drv_sub, color='red')
    axes[1].set_title("Driving Scores")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("driving_score")

    # (C) driving_score の差分推移
    axes[2].plot(ds_diff, color='green')
    axes[2].set_title("Derivative of Driving Scores")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("d(driving_score)/dt")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
