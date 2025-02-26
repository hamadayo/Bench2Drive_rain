import os
import re
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import numpy as np

def parse_locations_time_series(filepath):
    """
    ファイル内で複数行にわたり繰り返し出力される Location を時系列で取得する関数。
    例:
        Location: (x y z)
    の行をすべてパースし、[(x1,y1,z1), (x2,y2,z2), ...] というリストで返す。
    """
    location_pattern = re.compile(
        r'Location:\s*\((-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\)')
    locations = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            loc_match = location_pattern.search(line)
            if loc_match:
                x = float(loc_match.group(1))
                y = float(loc_match.group(2))
                z = float(loc_match.group(3))
                locations.append((x, y, z))

    return locations

def load_location_errors_for_all_files_with_offset(dirA, dirB, offset):
    """
    dirA の live_results_i.txt と dirB の live_results_(i - offset).txt を対応づけ、
    フレームごとのユークリッド距離誤差を計算して全部まとめて返す。
    
    例:
      - offset=622 の場合:
        A側: live_results_622.txt → B側: live_results_0.txt
        A側: live_results_623.txt → B側: live_results_1.txt
        ...
    """
    pattern = re.compile(r'^live_results_(\d+)\.txt$')

    # -- A側ファイル一覧を取得 --
    file_listA = []
    for fname in os.listdir(dirA):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))  # 622, 623, ...
            file_listA.append((idx, fname))
    file_listA.sort(key=lambda x: x[0])  # インデックスでソート

    # -- B側ファイル一覧を取得 --
    file_listB = []
    for fname in os.listdir(dirB):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))  # 0, 1, 2, ...
            file_listB.append((idx, fname))
    file_listB.sort(key=lambda x: x[0])

    # 辞書化しておくと対応参照がラク
    dictA = {idx: fname for idx, fname in file_listA}
    dictB = {idx: fname for idx, fname in file_listB}

    all_errors = []

    # -- A側のすべてのインデックス i について、B側は i - offset を探す --
    for i in dictA.keys():
        j = i - offset  # この j が B側のファイルインデックスとなる
        if j in dictB:
            # ファイルが両方存在するので位置情報を読み込み
            pathA = os.path.join(dirA, dictA[i])
            pathB = os.path.join(dirB, dictB[j])

            locsA = parse_locations_time_series(pathA)
            locsB = parse_locations_time_series(pathB)

            # フレームの少ない方に合わせて誤差計算
            common_len = min(len(locsA), len(locsB))
            for fidx in range(common_len):
                xA, yA, zA = locsA[fidx]
                xB, yB, zB = locsB[fidx]
                dist = np.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2)
                all_errors.append(dist)
        else:
            # j が存在しない → B側に対応するファイルがない
            pass

    return all_errors


def load_reliability_vals(text_file_path, max_lines=9999999):
    """
    新しいログ形式:
      [CAM_FRONT step=0000] reliability=1.0000
    のような行から数値を取り出してリストに格納する。
    """
    # 例: [CAM_FRONT step=0000] reliability=1.0000
    #     の中の reliability=1.0000 を取り出す正規表現
    pattern = re.compile(r'reliability=([\d\.]+)')

    reliability_vals = []
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        end_line = min(len(lines), max_lines)
        for i in range(end_line):
            line = lines[i].strip()
            match = pattern.search(line)
            if match:
                val_str = match.group(1)  # 1.0000 の部分
                try:
                    val_float = float(val_str)
                    reliability_vals.append(val_float)
                except ValueError:
                    # 数値変換できない場合はスキップ
                    pass
    return reliability_vals


def parse_locations_time_series(filepath):
    """
    ファイル内で複数行にわたり繰り返し出力される Location を時系列で取得する関数。
    例:
        Location: (x y z)
    の行をすべてパースし、[(x1,y1,z1), (x2,y2,z2), ...] というリストで返す。
    """
    location_pattern = re.compile(
        r'Location:\s*\((-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\)')
    locations = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            loc_match = location_pattern.search(line)
            if loc_match:
                x = float(loc_match.group(1))
                y = float(loc_match.group(2))
                z = float(loc_match.group(3))
                locations.append((x, y, z))

    return locations


def load_locations_time_series_in_dir(results_dir):
    """
    指定したディレクトリ内にある live_results_*.txt ファイルを
    ファイル名に含まれる数字順に読み込み、各ファイルの Location の時系列を
    すべて連結して返す。

    戻り値の例: [(x1, y1, z1), (x2, y2, z2), ..., (xN, yN, zN)]
    ただし、ファイル区切りなしに一つのリストにフレームとして並べる。
    """
    pattern = re.compile(r'^live_results_(\d+)\.txt$')
    file_list = []

    for fname in os.listdir(results_dir):
        match = pattern.match(fname)
        if match:
            file_index = int(match.group(1))
            file_list.append((file_index, fname))

    # 数値でソート
    file_list.sort(key=lambda x: x[0])

    all_locations = []
    for file_index, fname in file_list:
        path = os.path.join(results_dir, fname)
        locs = parse_locations_time_series(path)
        all_locations.extend(locs)  # 連結



    return all_locations


def load_location_errors_for_all_files(dirA, dirB):
    """
    2つのディレクトリ( dirA, dirB )内の live_results_*.txt を
    ファイル番号順にペアで比較し、フレームごとのユークリッド距離誤差を計算して
    全ファイルぶんを1つのリストに連結して返す。

    例:
      - dirA 内に live_results_0.txt, live_results_1.txt, ...
      - dirB 内にも 同じインデックスの live_results_0.txt, live_results_1.txt, ...
    のようにあると想定し、対応するファイル同士で誤差を計算する。
    """
    pattern = re.compile(r'^live_results_(\d+)\.txt$')

    # dirA, dirB それぞれで番号とファイル名を取得
    file_listA = []
    for fname in os.listdir(dirA):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            file_listA.append((idx, fname))
    file_listA.sort(key=lambda x: x[0])

    file_listB = []
    for fname in os.listdir(dirB):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            file_listB.append((idx, fname))
    file_listB.sort(key=lambda x: x[0])

    print(f"len(file_listA): {len(file_listA)}")
    print(f"len(file_listB): {len(file_listB)}")

    # 辞書化しておくと対応を探しやすい
    dictA = {idx: fname for idx, fname in file_listA}
    dictB = {idx: fname for idx, fname in file_listB}

    all_errors = []

    # 共通で存在するインデックスに対してフレームごとの誤差を計算・連結
    common_indices = sorted(set(dictA.keys()).intersection(dictB.keys()))
    for idx in common_indices:
        pathA = os.path.join(dirA, dictA[idx])
        pathB = os.path.join(dirB, dictB[idx])
        locsA = parse_locations_time_series(pathA)
        locsB = parse_locations_time_series(pathB)

        common_len = min(len(locsA), len(locsB))
        for i in range(common_len):
            xA, yA, zA = locsA[i]
            xB, yB, zB = locsB[i]
            dist = np.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2)
            all_errors.append(dist)

    return all_errors


def parse_file(filepath):
    """
    live_results_*.txt などから location (x, y, z) と driving_score を「最後に見つかったもの」だけ返す。
    例:
        Location: (-497.601 3672.927 364.864)
        Driving score: 0.000
    """
    x = y = z = None
    driving_score = None

    location_pattern = re.compile(
        r'Location:\s*\((-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\)')
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
    print(f'len(file_list): {len(file_list)}')

    all_driving_scores = []
    for file_index, filename in file_list:
        filepath = os.path.join(results_dir, filename)
        x, y, z, ds = parse_file(filepath)
        if ds is None:
            ds = 0.0
        all_driving_scores.append(ds)

    print(f'all_driving_scores: {all_driving_scores}')

    return all_driving_scores


def main():
    # -----------------------------------------------------
    # 例: 以下の2つのディレクトリを比較するとする
    # -----------------------------------------------------
    #   dirA = "/media/yoshi-22/T7/attention_no_rain/results"
    #   dirB = "/media/yoshi-22/T7/24211_reliability/results"
    # -----------------------------------------------------
    # --- 1. reliability_val の読み込み ---
    # reliability_file = "/media/yoshi-22/T7/24211_reliability/output_cluster_str/reliability/reliability_log_front.txt"
    reliability_file = "/home/yoshi-22/Bench2Drive/output_cluster_str/reliability/reliability_log_front.txt"
    
    reliability_vals = load_reliability_vals(reliability_file, max_lines=576)
    reliability_vals = reliability_vals[::5]

    print("len(reliability_vals) =", len(reliability_vals))


    # --- 2. driving_score の読み込み ---
    # results_dir = "/media/yoshi-22/T7/24211_reliability/results/0"  # live_results_*.txt があるディレクトリ
    results_dir = "/home/yoshi-22/Bench2Drive/output_cluster_str/results/0"
    driving_scores = load_driving_scores(results_dir)

    # --- 3. 軌跡の誤差を計算する ---
    # ここでは例として file1 と file2 の2つの結果ファイルを比較する想定。
    # 実際にはユーザーの状況に合わせてファイルパスを指定してください。
    # dirA = "/media/yoshi-22/T7/attention_no_rain/results/2"
    # dirB = "/media/yoshi-22/T7/24211_reliability/results/0"
    dirB = "/media/yoshi-22/T7/24211_reliability/results/0"
    dirA = "/home/yoshi-22/Bench2Drive/output_cluster_str/results/0"
    offset = 627

    # --- 2. 全ファイル分の driving_score の読み込み ---
    #     (例: dirB側のスコアにしたい場合など、必要に応じて変更)
    driving_scores = load_driving_scores(dirB)

    # --- 3. 全ファイル分の「位置誤差」を計算して取得 ---
    # location_errors = load_location_errors_for_all_files(dirA, dirB)
    location_errors = load_location_errors_for_all_files_with_offset(dirA, dirB, offset)

    print("len(location_errors)  =", len(location_errors))

    # =====================================================
    # ここから先は、driving_score 同様に "location_errors" を
    # フレーム時系列データとして扱い、相関やグラフ描画を行う例
    # =====================================================

    # --- 4. データの長さをそろえる（小さい方に合わせる） ---
    # （位置誤差とリライアビリティはフレームごとの大きい系列、
    # 　一方、driving_scores は「ファイル数の系列」なので粒度が違う）
    common_length = min(len(reliability_vals), len(location_errors))
    rel_sub = reliability_vals[:common_length]
    loc_err_sub = location_errors[:common_length]

    # -----------------------------------------------------
    # 注意: driving_scores の長さは「ファイル数」になるので、
    #       もし 1ファイル=1フレーム と見なすなら len(driving_scores) と
    #       reliability_vals[:len(driving_scores)] を比較できるかもしれません。
    #       しかし実際にはフレーム粒度とは異なるため、そのまま相関を計算すると
    #       次元が合わなくなる点に注意してください。
    # -----------------------------------------------------

    # 例として、フレーム粒度の "loc_err_sub" と "reliability" の相関だけをとる
    loc_err_diff = np.diff(loc_err_sub)
    rel_sub_diff = rel_sub[1:]  # 1つシフト

    common_length_diff = min(len(loc_err_diff), len(rel_sub_diff))
    loc_err_diff = loc_err_diff[:common_length_diff]
    rel_sub_diff = rel_sub_diff[:common_length_diff]

    if common_length_diff > 1:
        corr_mat = np.corrcoef(loc_err_diff, rel_sub_diff)
        corr_coeff = corr_mat[0, 1]
        print(f"Corr(reliability_diff, location_error_diff) = {corr_coeff:.4f}")
    else:
        print("有効なデータ点がほとんどないため、誤差とリライアビリティの微分の相関係数を計算できません。")

    # --- 5. グラフ描画 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (A) reliability_val の推移（フレーム粒度）
    axes[0].plot(rel_sub, color='blue')
    axes[0].set_title("Reliability Values (frame-based)")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("reliability_val")
    axes[0].set_xlim(45, 55)
    axes[0].axvline(x=46, color='k', linestyle='--')
    axes[0].axvline(x=51, color='k', linestyle='--')

    # (B) location_error の推移（フレーム粒度）
    # axes[1].plot(loc_err_sub, color='purple')
    # axes[1].set_title("Location Errors (frame-based)")
    # axes[1].set_xlabel("Frame")
    # axes[1].set_ylabel("Error distance")
    # axes[1].set_xlim(0, 100)
    # axes[1].axvline(x=47, color='k', linestyle='--')
    # axes[1].axvline(x=51, color='k', linestyle='--')

    # (C) driving_score はファイル単位なのでフレームとは数が一致しない。
    #     例として「ファイルインデックス vs. ドライビングスコア」を表示
    axes[2].plot(driving_scores, color='red', marker='o')
    axes[2].set_title("Driving Scores (file-based)")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("driving_score")
    axes[2].set_xlim(45, 55)
    # axes[2].axvline(x=50, color='k', linestyle='--')
    axes[2].axvline(x=51, color='k', linestyle='--')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
