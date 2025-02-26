import numpy as np
import matplotlib.pyplot as plt
import os
import random

from scipy.fftpack import fft2
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import tensorflow as tf

# GPUの利用確認
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"利用可能なGPU: {gpus}")
    try:
        # GPUメモリの動的割り当てを有効にする
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("GPUが見つかりません。CPUを使用します。")


# -----------------------------------
# 定数定義
# -----------------------------------
IMAGE_WIDTH, IMAGE_HEIGHT = 900, 512   # 画像サイズ
ROI_SIZE = 49                         # ROI (Region of Interest) のサイズ
R_VALUES = list(range(1, 15, 1)) + list(range(18, 21, 1))  # 半径範囲
GABOR_R = [2, 3, 4, 5]
GABOR_THETA = [i for i in range(20, 90, 5)]

# MIN_CAPACITY = 20   # データ生成の最小容量
# MAX_CAPACITY = 130  # データ生成の最大容量
# MAX_GENERATIONS = 20  # 個体の最大世代数
# FITNESS_THRESHOLD = 0.6
# INITIAL_POPULATION_SIZE = 60
# X_SPACING = 20  # グリッドの横方向間隔
# Y_SPACING = 20  # グリッドの縦方向間隔
# OFFSET_RANGE = 40  # 親近傍での候補点ずらし範囲
# CONFLICT_THRESHOLD = 20  # 衝突判定の閾値
# DISTANCE_THRESHOLD = 50  # クラスタ重心のマージ判定距離
# EPS = 60  # DBSCANのepsパラメータ
# MIN_CLUSTER_SIZE = 5  # クラスタリングの最小サンプル数

MIN_CAPACITY = 20   # データ生成の最小容量
MAX_CAPACITY = 200  # データ生成の最大容量
MAX_GENERATIONS = 20  # 個体の最大世代数
FITNESS_THRESHOLD = 0.6
INITIAL_POPULATION_SIZE = 90
X_SPACING = 10 # グリッドの横方向間隔
Y_SPACING = 10  # グリッドの縦方向間隔
OFFSET_RANGE = 30  # 親近傍での候補点ずらし範囲
CONFLICT_THRESHOLD = 20  # 衝突判定の閾値
DISTANCE_THRESHOLD = 50  # クラスタ重心のマージ判定距離
EPS = 40  # DBSCANの距離閾値
MIN_CLUSTER_SIZE = 4


# ====================================================
# (1) グリッド状候補点の生成
# ====================================================
def generate_grid_points(width, height, roi_size, spacing_x=5, spacing_y=5):
    """
    画像内でROIを置ける位置のうち、spacing_x, spacing_y で区切った
    グリッド交点を候補点リストとして返す。
    """
    candidate_points = []
    for y in range(0, height - roi_size + 1, spacing_y):
        for x in range(0, width - roi_size + 1, spacing_x):
            candidate_points.append((x, y))
    
    print(f"Generated {len(candidate_points)} candidate points.")
    return candidate_points

# -----------------------------------------------------
# (2)-A) 親個体近傍での候補点取得
# -----------------------------------------------------
def pick_random_grid_near_parent(px, py,
                                 image_width, image_height, roi_size,
                                 existing_individuals,
                                 offset_range=5,
                                 max_tries=30):
    """
    親個体(px, py)近傍 offset_rangeピクセル以内からランダムに点を選んで
    グリッド(間隔X_SPACING, Y_SPACING)に丸めて返す。
    既存個体と衝突しない位置を見つければタプル(x, y)を返す。失敗時None。
    """
    for _ in range(max_tries):
        dx = random.randint(-offset_range, offset_range)
        dy = random.randint(-offset_range, offset_range)
        
        x_cand = px + dx
        y_cand = py + dy
        
        # ROIが画像外にはみ出ないようクリップ
        x_cand = max(0, min(x_cand, image_width - roi_size))
        y_cand = max(0, min(y_cand, image_height - roi_size))
        
        # グリッドに丸める
        x_cand = (x_cand // X_SPACING) * X_SPACING
        y_cand = (y_cand // Y_SPACING) * Y_SPACING
        
        # 衝突判定
        if not conflict_with_existing(x_cand, y_cand, existing_individuals, roi_size):
            return (x_cand, y_cand)
    return None

# -----------------------------------------------------
# (2)-B) 完全ランダムに点を取得（グリッドに丸める）
# -----------------------------------------------------
def pick_random_grid_point(image_width, image_height, roi_size,
                           existing_individuals,
                           max_tries=30):
    """
    画像の中で(間隔X_SPACING, Y_SPACING)のグリッド上からランダムに点を選ぶ。
    既存個体と衝突しなければ(x, y)を返す。失敗時None。
    """
    for _ in range(max_tries):
        max_x_grid = (image_width - roi_size) // X_SPACING
        max_y_grid = (image_height - roi_size) // Y_SPACING
        
        gx = random.randint(0, max_x_grid)
        gy = random.randint(0, max_y_grid)
        
        x_cand = gx * X_SPACING
        y_cand = gy * Y_SPACING
        
        if not conflict_with_existing(x_cand, y_cand, existing_individuals, roi_size):
            return (x_cand, y_cand)
    return None

# -----------------------------------------------------
# (2)-C) 衝突判定
# -----------------------------------------------------
def conflict_with_existing(x_cand, y_cand, existing_points, roi_size):
    """
    (x_cand, y_cand)が既存の個体と近すぎるかチェック
    """
    for indiv in existing_points:
        dx = abs(x_cand - indiv['x'])
        dy = abs(y_cand - indiv['y'])
        if dx <= CONFLICT_THRESHOLD and dy <= CONFLICT_THRESHOLD:
            return True
    return False

# -----------------------------------------------------
# (3) 円形マスク作成 (PSD用)
# -----------------------------------------------------
def create_circular_mask(width, height, radius, upper_half_only=False):
    """
    円形/半円マスクを作成。中心からradius±0.5ピクセルの領域をTrueとする。
    upper_half_only=Trueの場合、画像上半分に限定。
    """
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist_from_center >= radius - 0.5) & (dist_from_center < radius + 0.5)
    if upper_half_only:
        mask &= (y <= center_y)
        mask &= (x <= center_x)
    return mask

# -----------------------------------------------------
# (4) 周波数ドメインのガボールフィルタ作成 (例)
# -----------------------------------------------------
def create_freq_gabor_filter_bank(roi_size=49, radii=[3,6,12], angles=range(0,180,5)):
    """
    周波数ドメインで輪状ガウシアンに近いガボールフィルタ（の簡易版）を複数作成。
    """
    filters = []
    U, V = np.mgrid[0:roi_size, 0:roi_size]
    U_shift = U - roi_size // 2
    V_shift = V - roi_size // 2

    for r in radii:
        for theta in angles:
            angle_rad = np.deg2rad(theta)
            Ur = U_shift * np.cos(angle_rad) + V_shift * np.sin(angle_rad)
            Vr = -U_shift * np.sin(angle_rad) + V_shift * np.cos(angle_rad)
            
            sigma = r / 2
            gb = np.exp(-0.5 * ((Ur)**2 + (Vr - r)**2) / sigma**2) * np.cos(2 * np.pi * r * Ur / roi_size)
            norm_val = np.linalg.norm(gb)
            if norm_val > 1e-9:
                gb /= norm_val
            filters.append(gb)
    
    return filters

# -----------------------------------------------------
# (5) PSD計算 + ガボール応答抽出
# -----------------------------------------------------
def calculate_psd_features(roi, freq_gabor_filters, num_accumulate=5):
    """
    1) ログ変換 + ガウシアン + Hanning窓でFFT
    2) PSDを num_accumulate 回累積
    3) 周波数ガボールフィルタ適用
    4) 半径rごとの積分 + 2次多項式フィット係数
    5) 輝度統計などをまとめて特徴量化
    """
    roi_float = roi.astype(np.float32)
    accumulated_psd = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32)

    for _ in range(num_accumulate):
        roi_log = np.log1p(roi_float)
        roi_filt = gaussian_filter(roi_log, sigma=1.0)

        # 2Dハニング窓
        hanning_1d = np.hanning(ROI_SIZE)
        hanning_2d = np.outer(hanning_1d, hanning_1d)
        roi_hanned = roi_filt * hanning_2d

        fft_roi = fft2(roi_hanned)
        psd = np.abs(fft_roi)**2

        accumulated_psd += psd

    # 周波数ドメインガボールフィルタ適用 (総和を取る)
    gabor_responses = []
    for gf in freq_gabor_filters:
        response_val = np.sum(accumulated_psd * gf)
        gabor_responses.append(response_val)

    # 円形マスクで半径ごとの積分
    psd_sums = []
    for r in R_VALUES:
        mask = create_circular_mask(ROI_SIZE, ROI_SIZE, r, upper_half_only=True)
        psd_sums.append(np.sum(accumulated_psd[mask]))
    psd_sums = np.array(psd_sums, dtype=np.float32)

    # 正規化
    psd_max = np.max(psd_sums)
    psd_min = np.min(psd_sums)
    if psd_max - psd_min > 1e-9:
        psd_sums = (psd_sums - psd_min)/(psd_max - psd_min)
    psd_sums = psd_sums.tolist()

    # 2次多項式フィット
    xvals = np.array(R_VALUES, dtype=np.float32)
    yvals = np.array(psd_sums, dtype=np.float32)
    X_fit = np.vstack((xvals**2, xvals)).T
    linreg = LinearRegression().fit(X_fit, yvals)
    polyA = linreg.coef_[0]
    polyB = linreg.coef_[1]
    polyC = linreg.intercept_

    # ROI輝度統計（最大値、最小値、標準偏差）
    max_val = float(np.max(roi_float))
    min_val = float(np.min(roi_float))
    std_val = float(np.std(roi_float))
    bright_diff = max_val - min_val

    # 特徴ベクトルを連結
    features = (
        list(gabor_responses)
        + list(psd_sums)
        + [polyA, polyB, polyC, bright_diff, std_val]
    )
    return features

# -----------------------------------------------------
# (6) 初期個体群の作成
# -----------------------------------------------------
def initialize_population(size):
    population = []
    existing_list = []
    for _ in range(size):
        point = pick_random_grid_point(IMAGE_WIDTH, IMAGE_HEIGHT, ROI_SIZE, existing_list)
        if point is None:
            continue
        x_cand, y_cand = point
        ind = {
            "x": x_cand,
            "y": y_cand,
            "fitness": 0.0,
            "age": 0,
            "max_age": MAX_GENERATIONS,
        }
        population.append(ind)
        existing_list.append(ind)
    return population

# -----------------------------------------------------
# (7) 容量調整
# -----------------------------------------------------
def adjust_carrying_capacity(current_capacity, survived_ratio):
    """
    生存率に応じてcarrying_capacityを増減する簡単な例
    """
    if survived_ratio < 0.2 and current_capacity > MIN_CAPACITY:
        return current_capacity - 10
    elif survived_ratio > 0.6 and current_capacity < MAX_CAPACITY:
        return current_capacity + 10
    print(f"  Survived ratio: {survived_ratio:.2f}, Current carrying capacity: {current_capacity}")
    return current_capacity

# -----------------------------------------------------
# (8) 個体選択
# -----------------------------------------------------
def select_population(population, carrying_capacity):
    # fitnessで足切り
    selected = [ind for ind in population if ind["fitness"] >= FITNESS_THRESHOLD]
    # age制限
    selected = [ind for ind in selected if ind["age"] < ind["max_age"]]
    # fitness順にソートして上位を残す
    selected.sort(key=lambda i: i["fitness"], reverse=True)
    return selected[:carrying_capacity]

# -----------------------------------------------------
# (9) 近接クラスタのマージ
# -----------------------------------------------------
def merge_close_clusters(population, labels, distance_threshold=50):
    """
    DBSCANのクラスタラベルを元に、重心が近いクラスタ同士をマージして再ラベリング。
    """
    if len(population) == 0 or len(labels) == 0:
        return labels

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    unique_labels = sorted(list(unique_labels))

    if not unique_labels:
        return labels

    # 各クラスタ重心
    centroids = {}
    for lab in unique_labels:
        coords = np.array([
            [p["x"], p["y"]] 
            for p, lab_ in zip(population, labels) if lab_ == lab
        ])
        centroid = coords.mean(axis=0)
        centroids[lab] = centroid

    adjacency = {lab: [] for lab in unique_labels}

    def euclid_dist(a, b):
        return np.linalg.norm(a - b)

    # クラスタ同士の距離がthreshold未満なら隣接とみなす
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            c1 = unique_labels[i]
            c2 = unique_labels[j]
            dist_ = euclid_dist(centroids[c1], centroids[c2])
            if dist_ < distance_threshold:
                adjacency[c1].append(c2)
                adjacency[c2].append(c1)

    visited = set()
    new_label_map = {}
    merged_label_id = 0

    def bfs(start):
        queue = [start]
        group = []
        visited.add(start)
        while queue:
            current = queue.pop(0)
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return group

    # BFSで近隣クラスタをまとめる
    for lab in unique_labels:
        if lab not in visited:
            group = bfs(lab)
            for g in group:
                new_label_map[g] = merged_label_id
            merged_label_id += 1

    # 個体ごとのラベルを置き換え
    new_labels = []
    for lab in labels:
        if lab == -1:
            new_labels.append(-1)
        else:
            new_labels.append(new_label_map[lab])
    return np.array(new_labels)

# -----------------------------------------------------
# (10) 繁殖
# -----------------------------------------------------
def reproduce_population(selected_population, total_size, candidate_points, roi_size, ratio_parent=0.7):
    """
    親近傍に子を生む: ratio_parent
    完全ランダムに子を生む: 1-ratio_parent
    """
    new_population = []
    n_offsprings = total_size - len(selected_population)
    existing_list = new_population + selected_population

    if len(selected_population) == 0:
        # 親が居ない場合は全てランダム
        for _ in range(n_offsprings):
            point = pick_random_grid_point(IMAGE_WIDTH, IMAGE_HEIGHT, roi_size, existing_list)
            if point is None:
                continue
            x_cand, y_cand = point
            ind = {
                "x": x_cand,
                "y": y_cand,
                "fitness": 0.0,
                "age": 0,
                "max_age": np.random.randint(5, MAX_GENERATIONS),
            }
            new_population.append(ind)
            existing_list.append(ind)
        return new_population

    n_parent_based = int(n_offsprings * ratio_parent)
    n_random = n_offsprings - n_parent_based

    if n_parent_based > len(selected_population):
        # 必要数が親数を超えるならランダムに親を重複選抜
        parent_list = [random.choice(selected_population) for _ in range(n_parent_based)]
    else:
        parent_list = random.sample(selected_population, n_parent_based)
    
    # --- (1) 親近傍オフスプリング ---
    for parent in parent_list:
        point = pick_random_grid_near_parent(
            px=parent["x"],
            py=parent["y"],
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            roi_size=roi_size,
            existing_individuals=existing_list,
            offset_range=OFFSET_RANGE,
            max_tries=30
        )
        if point:
            x_cand, y_cand = point
            offspring = {
                "x": x_cand,
                "y": y_cand,
                "fitness": 0.0,
                "age": 0,
                "max_age": np.random.randint(5, MAX_GENERATIONS),
            }
            new_population.append(offspring)
            existing_list.append(offspring)

    # --- (2) ランダムオフスプリング ---
    for _ in range(n_random):
        point = pick_random_grid_point(IMAGE_WIDTH, IMAGE_HEIGHT, roi_size, existing_list, max_tries=30)
        if point is None:
            break
        x_cand, y_cand = point
        offspring = {
            "x": x_cand,
            "y": y_cand,
            "fitness": 0.0,
            "age": 0,
            "max_age": np.random.randint(5, MAX_GENERATIONS),
        }
        new_population.append(offspring)
        existing_list.append(offspring)
        
    # 親世代は年齢 +1
    for ind in selected_population:
        ind["age"] += 1
    
    # 親を次世代へ合流
    new_population.extend(selected_population)
    return new_population

# -----------------------------------------------------
# (11) 個体のfitness計算（モデル推定）
# -----------------------------------------------------
def evaluate_population(population, image, freq_gabor_filters, loaded_model, selector):
    """
    各個体のROIを切り出し -> PSD特徴抽出 -> 特徴選択(selector) -> 推論モデルに通す
    """
    for individual in population:
        x, y = individual["x"], individual["y"]
        roi = image[y:y+ROI_SIZE, x:x+ROI_SIZE]

        features = calculate_psd_features(roi, freq_gabor_filters, num_accumulate=5)
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)

        # 事前にANOVA等で学習した特徴選択器を適用
        features_array_60 = selector.transform(features_array)

        # 推定結果をfitnessとして格納
        fitness_val = loaded_model.predict(features_array_60)[0, 0]
        individual["fitness"] = float(fitness_val)

# -----------------------------------------------------
# (12) DBSCANなどでクラスタリング
# -----------------------------------------------------
def cluster_population(population, eps=60, min_samples=5):
    if not population:
        return []
    coords = np.array([[ind["x"], ind["y"]] for ind in population])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = merge_close_clusters(population, db.labels_, distance_threshold=DISTANCE_THRESHOLD)
    
    return labels



# def visualize_clusters(population, labels, img_array, title="Clusters", out_dir="./out_b2d"):
#     """
#     ConvexHullを描いてクラスタ可視化。クラスタの面積などを表示。
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     plt.figure()
#     unique_labels = set(labels)
#     colors = plt.cm.get_cmap("tab20", len(unique_labels))
#     cluster_area_texts = []

#     for idx, k in enumerate(unique_labels):
#         class_member_mask = (labels == k)
#         roi_corners = []

#         for ind, is_member in zip(population, class_member_mask):
#             if not is_member:
#                 continue
#             x0, y0 = ind["x"], ind["y"]
#             corners = [
#                 (x0, y0),
#                 (x0 + ROI_SIZE, y0),
#                 (x0 + ROI_SIZE, y0 + ROI_SIZE),
#                 (x0, y0 + ROI_SIZE)
#             ]
#             roi_corners.extend(corners)

#         if len(roi_corners) == 0:
#             continue

#         xy = np.array(roi_corners)
#         xy2 = np.array([[p["x"], p["y"]] for p, m in zip(population, class_member_mask) if m])

#         if k == -1:
#             plt.scatter(xy2[:,0], xy2[:,1], c="k", marker="x", label=None)
#         else:
#             color_val = colors(idx)
#             plt.scatter(xy2[:,0], xy2[:,1], marker="o", color=color_val, label=None)

#             if len(xy) > 3:
#                 hull = ConvexHull(xy)
#                 area = hull.area
#                 hull_vertices = hull.vertices
#                 hull_points = xy[hull_vertices]
#                 print(f"  Cluster {k}: {len(xy)} points, area={area:.2f}")
#                 plt.fill(hull_points[:,0], hull_points[:,1], color=color_val, alpha=0.3)
#                 cluster_area_texts.append(f"C{k}={area:.2f}")

    # if cluster_area_texts:
    #     area_text = ", ".join(cluster_area_texts)
    #     plt.title(f"{title} ({area_text})")
    # else:
    #     plt.title(title)

    # plt.grid(False)
    # plt.imshow(img_array, cmap="gray", alpha=0.5)
    # save_path = os.path.join(out_dir, f"{title}.png")
    # plt.savefig(save_path)
    # plt.close()

def visualize_clusters(
    population, labels, img_array,
    title="Clusters", out_dir="./out_b2d", roi_size=49
):
    """
    クラスタ全体の外枠（ConvexHull）を描画せず、
    各個体の ROI 枠 (Rectangle) のみを重ね描画する関数。
    """
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    # クラスタ数に応じたカラーマップ
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    # 画像を背景に表示
    plt.imshow(img_array, cmap="gray", alpha=0.5)

    for idx, k in enumerate(unique_labels):
        class_member_mask = (labels == k)
        
        # クラスタ k の点 (個体の (x,y))
        xy_cluster = np.array([
            [p["x"], p["y"]] 
            for p, is_member in zip(population, class_member_mask) 
            if is_member
        ])

        if len(xy_cluster) == 0:
            continue
        
        # 色の設定
        if k == -1:
            # -1 はノイズクラスタ
            color_val = "k"
            plt.scatter(xy_cluster[:, 0], xy_cluster[:, 1],
                        c=color_val, marker="x", label=None)
        else:
            color_val = colors(idx)
            plt.scatter(xy_cluster[:, 0], xy_cluster[:, 1],
                        c=[color_val], marker="o", label=None)

        # === 各個体の ROI 枠を描画 ===
        for indiv, is_member in zip(population, class_member_mask):
            if not is_member:
                continue
            x0, y0 = indiv["x"], indiv["y"]
            rect = patches.Rectangle(
                (x0, y0),        # 左上座標
                roi_size,        # 幅
                roi_size,        # 高さ
                linewidth=2,
                edgecolor=color_val,
                facecolor='none'
            )
            plt.gca().add_patch(rect)

    plt.title(title)
    save_path = os.path.join(out_dir, f"{title}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure: {save_path}")


def ea_process_single_frame(population, carrying_capacity,
                            gray_image_np, loaded_model, selector,
                            freq_gabor_filters, candidate_points, step, out_dir):
    """
    population, carrying_capacity を更新して返す
    - 1フレーム分のEAステップ
    - DBSCAN→近接クラスタマージ→凸包面積なども計算
    戻り値:
      updated_population,
      updated_carrying_capacity,
      labels (クラスタラベル配列)
    """
    # (A) 個体を評価
    evaluate_population(population, gray_image_np, freq_gabor_filters, loaded_model, selector)

    # (B) クラスタリング
    labels = cluster_population(population, eps=EPS, min_samples=MIN_CLUSTER_SIZE)
    visualize_clusters(population, labels, gray_image_np, f"Clusters_{step}", out_dir)

    # (C) 選択
    survived = select_population(population, carrying_capacity)
    ratio = len(survived)/float(len(population)) if len(population)>0 else 0

    # (D) carrying capacity 調整
    carrying_capacity = adjust_carrying_capacity(carrying_capacity, ratio)

    # (E) 繁殖
    #   注意: reproduce_populationには candidate_points が必要。
    new_population = reproduce_population(survived, carrying_capacity,
                                          candidate_points, ROI_SIZE)

    return new_population, carrying_capacity, labels

