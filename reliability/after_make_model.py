import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIL import Image
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
import cv2
import random
from sklearn.cluster import DBSCAN
from multiprocessing import Pool, cpu_count
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import glob
from multiprocessing import Pool, cpu_count
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
import joblib  # または pickle でも可
from scipy.spatial import ConvexHull


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

# 定数定義
IMAGE_WIDTH, IMAGE_HEIGHT = 900, 512  # 画像サイズ
ROI_SIZE = 49  # ROI (Region of Interest) のサイズ
# R_VALUES = list(range(1, 15, 1)) + list(range(18, 21, 1))  # 半径範囲
# GABOR_R = [2, 3, 4, 5]
# GABOR_THETA = [i for i in range(20, 90, 5)]
R_VALUES = range(5, 25, 4)  # 半径範囲
GABOR_R = [i for i in range(1, 25)]
GABOR_THETA = [i for i in range(0, 180, 5)]  # ガボール方向

MIN_CAPACITY = 20  # データ生成の最小容量
MAX_CAPACITY = 130  # データ生成の最大容量
MAX_GENERATIONS = 20  # 最大世代数
FITNESS_THRESHOLD = 0.6
INITIAL_POPULATION_SIZE = 60
X_SPACING = 20 # グリッドの間隔
Y_SPACING = 20 # グリッドの間隔
OFFSET_RANGE = 40 # ランダムオフセットの範囲
CONFLICT_THRESHOLD = 20  # 衝突判定の閾値
DISTANCE_THRESHOLD = 50  # クラスタリングの距離閾値
EPS = 60  # DBSCANの距離パラメータ
MIN_CLUSTER_SIZE = 5  # クラスタリングの最小クラスタサイズ

import numpy as np
import matplotlib.pyplot as plt




# -----------------------------
# 1) グリッド状の候補点を生成する関数
# -----------------------------
def generate_grid_points(width, height, roi_size, spacing_x=5, spacing_y=5):
    """
    画像内で ROI_SIZE×ROI_SIZE を置ける位置のうち、縦・横の線間隔が spacing_x, spacing_y となるような
    グリッドの交点候補をあらかじめ生成してリストで返す。

    Args:
        image_width (int): 画像の幅
        image_height (int): 画像の高さ
        roi_size (int): ROIのサイズ(正方形想定)
        spacing_x (int): 横方向の線の間隔
        spacing_y (int): 縦方向の線の間隔

    Returns:
        list of tuple: 候補点 (x, y) のリスト
    """
    candidate_points = []
    for y in range(0, height - roi_size + 1, spacing_y):
        for x in range(0, width - roi_size + 1, spacing_x):
            candidate_points.append((x, y))
    
    print(f"Generated {len(candidate_points)} candidate points.")
    return candidate_points

# -----------------------------------------------------
# B) 親個体近くのランダム点を選ぶ関数（グリッドに丸める）
# -----------------------------------------------------
def pick_random_grid_near_parent(px, py,
                                 image_width, image_height, roi_size,
                                 existing_individuals,
                                 offset_range=5,
                                 max_tries=30):
    """
    親個体 (px, py) の周囲 offset_range ピクセル以内で
    ランダムに点を選んで 5の倍数座標に丸める。
    衝突しなければ (x_cand, y_cand) を返す。失敗なら None。
    """
    for _ in range(max_tries):
        # 親近傍: +/- offset_range の範囲で乱数
        dx = random.randint(-offset_range, offset_range)
        dy = random.randint(-offset_range, offset_range)
        
        x_cand = px + dx
        y_cand = py + dy
        
        # ROIが画像外にはみ出ないようクリップ
        x_cand = max(0, min(x_cand, image_width - roi_size))
        y_cand = max(0, min(y_cand, image_height - roi_size))
        
        # 5ピクセルの倍数に丸める
        x_cand = (x_cand // X_SPACING) * X_SPACING
        y_cand = (y_cand // Y_SPACING) * Y_SPACING
        
        # 衝突判定
        if not conflict_with_existing(x_cand, y_cand, existing_individuals, roi_size):
            return (x_cand, y_cand)
    return None

# -----------------------------------------------------
# C) 完全ランダムに点を選ぶ関数（グリッドに丸める）
# -----------------------------------------------------
def pick_random_grid_point(image_width, image_height, roi_size,
                           existing_individuals,
                           max_tries=30):
    """
    画像の中で 5ピクセル刻みに丸めた点をランダムに選び、
    衝突しなければ (x_cand, y_cand) を返す。失敗なら None。
    """
    for _ in range(max_tries):
        # (image_width - roi_size) // 5 が作れる最大分割数
        max_x_grid = (image_width - roi_size) // X_SPACING
        max_y_grid = (image_height - roi_size) // Y_SPACING
        
        # その範囲でランダムに「グリッド何マス目か」を選ぶ
        gx = random.randint(0, max_x_grid)
        gy = random.randint(0, max_y_grid)
        
        # 実際の画素座標に変換
        x_cand = gx * X_SPACING
        y_cand = gy * Y_SPACING
        
        # 衝突判定
        if not conflict_with_existing(x_cand, y_cand, existing_individuals, roi_size):
            return (x_cand, y_cand)
    return None


def conflict_with_existing(x_cand, y_cand, existing_points, roi_size):
    """
    既存の点と衝突するかどうかを判定する関数。

    Args:
        x_cand (int): 候補点の x 座標
        y_cand (int): 候補点の y 座標
        existing_points (list of tuple): すでに存在する点のリスト
        roi_size (int): ROIのサイズ

    Returns:
        bool: 衝突する場合は True、しない場合は False
    """
    for indiv in existing_points:
        dx = abs(x_cand - indiv['x'])
        dy = abs(y_cand - indiv['y'])
        if dx <= CONFLICT_THRESHOLD and dy <= CONFLICT_THRESHOLD:
            return True
        else:
            print(f"  Conflict check: ({x_cand}, {y_cand}) vs ({indiv['x']}, {indiv['y']})")
    return False


def visualize_selected_features(selector, feature_names=None):
    """
    選択された特徴量の F値、p値 をバーグラフで可視化する関数。
    
    Args:
        selector: SelectKBestなどの特徴選択オブジェクト (fit済み)。
        feature_names (list or None): 元の特徴量名のリスト。Noneの場合はインデックスを使用。
    """
    # 各特徴量の F値, p値
    f_values = selector.scores_
    p_values = selector.pvalues_
    
    # 選ばれた特徴量のブール配列 (True/False)
    selected_mask = selector.get_support()
    
    # 選ばれた特徴量だけを抽出
    selected_f_values = f_values[selected_mask]
    selected_p_values = p_values[selected_mask]
    
    # 特徴量名が指定されていればそれも対応して取り出す。なければインデックス番号で代用
    if feature_names is not None:
        selected_feature_names = np.array(feature_names)[selected_mask]
    else:
        # Feature i という名前を自動でつける例
        all_indices = np.array([f"Feature {i}" for i in range(len(f_values))])
        selected_feature_names = all_indices[selected_mask]
    
    # ===== 可視化：F値 =====
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(selected_f_values))
    plt.bar(x_pos, selected_f_values, alpha=0.7, color='blue')
    plt.xticks(x_pos, selected_feature_names, rotation=90)
    plt.title("Selected Features by ANOVA F-value")
    plt.xlabel("Features")
    plt.ylabel("F-value")
    plt.tight_layout()
    plt.show()
    
    # ===== 可視化：p値 =====
    plt.figure(figsize=(10, 6))
    # -log10(p) などでスケールを取りやすくする例
    neg_log_p_values = -np.log10(selected_p_values + 1e-15)
    plt.bar(x_pos, neg_log_p_values, alpha=0.7, color='red')
    plt.xticks(x_pos, selected_feature_names, rotation=90)
    plt.title("Selected Features by p-value (shown as -log10(p))")
    plt.xlabel("Features")
    plt.ylabel("-log10(p-value)")
    plt.tight_layout()
    plt.show()


def plot_label_distribution(y):
    plt.hist(y, bins=50, color='blue', edgecolor='black')
    plt.title('Label Distribution')
    plt.xlabel('Average Label Value')
    plt.ylabel('Frequency')
    plt.show()

def create_freq_gabor_filter_bank(roi_size=49, radii=[3,6,12,18,21], angles=range(0, 180, 5)):
    """
    周波数ドメインのガボールフィルタを簡易的に作る
    """
    filters = []
    U, V = np.mgrid[0:roi_size, 0:roi_size]
    U_shift = U - roi_size // 2
    V_shift = V - roi_size // 2
    sigma = roi_size / 6

    for r in radii:
        for theta in angles:
            # ガボールの中心周波数などを決定
            # freq: 半径r付近を中心とする簡易ガウシアン
            # theta: 角度方向を回転
            # sigmaの値などは論文に合わせて要調整
            sigma = r / 2
            angle_rad = np.deg2rad(theta)

            # 回転後の座標 (Ur, Vr)
            Ur = U_shift * np.cos(angle_rad) + V_shift * np.sin(angle_rad)
            Vr = -U_shift * np.sin(angle_rad) + V_shift * np.cos(angle_rad)

            # ガウシアン
            gb = np.exp(-0.5 * ((Ur)**2 + (Vr - r)**2) / sigma**2) * np.cos(2 * np.pi * r * Ur / roi_size)
            norm_val = np.linalg.norm(gb)
            if norm_val > 1e-9:
                gb /= norm_val

            # 位相シフト等が必要なら追加で掛け合わせる
            # ここでは振動成分を省いて "輪状ガウシアン" としている
            # 実際のガボールは cos( ... ) などを掛けることが多い
            filters.append(gb)
    
    return filters

def create_circular_mask(width, height, radius, upper_half_only=False):
    """
    円形マスクまたは半円マスクを作成する関数。

    Args:
        width (int): マスクの幅（画像サイズに対応）。
        height (int): マスクの高さ（画像サイズに対応）。
        radius (int): 円の半径（ちょうど一致する領域を選択）。
        upper_half_only (bool): True の場合、画像上半分のみにマスクを制限。

    Returns:
        np.ndarray: 指定した半径と条件に合致するマスク（真偽値の2D配列）。
    """
    # 画像の中心座標
    center_x, center_y = width // 2, height // 2

    # 各ピクセルの座標を生成
    y, x = np.ogrid[:height, :width]

    # 中心からの距離を計算
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # 半径がちょうど一致する領域を選択
    mask = (distance_from_center >= radius - 0.5) & (distance_from_center < radius + 0.5)
    if upper_half_only:
        mask &= (y <= center_y )
        mask &= (x <= center_x )

    return mask

##############################################################################
# (2) PSD計算 (Hanning窓 + 5回累積 + 周波数ドメインガボール)
##############################################################################
def calculate_psd_features(roi, freq_gabor_filters, num_accumulate=5):
    """
    PSDを num_accumulate回 累積し、最後に周波数ガボールを適用。
    """
    # 0. ROIをfloat化 & ログ変換 & ガウシアン
    roi_float = roi.astype(np.float32)

    # 簡易的に 5回同じROIを用いる例: 実際は「連続5フレームのROI」を合算するなど工夫
    accumulated_psd = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32)

    for _ in range(num_accumulate):
        # 1) ログ変換
        roi_log = np.log1p(roi_float)
        # 2) ガウシアン
        roi_filt = gaussian_filter(roi_log, sigma=1.0)
        # 3) 2Dハニング窓
        hanning_1d = np.hanning(ROI_SIZE)
        hanning_2d = np.outer(hanning_1d, hanning_1d)
        roi_hanned = roi_filt * hanning_2d

        # 4) FFT & PSD
        fft_roi = fft2(roi_hanned)
        psd = np.abs(fft_roi)**2

        accumulated_psd += psd

    # num_accumulate回の合計を使う
    # 必要に応じて平均にしても良い: accumulated_psd /= num_accumulate

    # (A) 周波数ドメインのガボールフィルタを適用して合計値を取る
    # filters: freq_gabor_filters (list of 2D array)
    gabor_responses = []
    for gf in freq_gabor_filters:
        # gf と PSDを乗算して、総和を取る
        # gf.shape, psd.shape = (ROI_SIZE, ROI_SIZE)
        response_val = np.sum(accumulated_psd * gf)
        gabor_responses.append(response_val)

    # (B) PSD-summation features
    # rごとに積分 (例)
    psd_sums = []
    for r in R_VALUES:
        mask = create_circular_mask(ROI_SIZE, ROI_SIZE, r, upper_half_only=True)
        psd_sums.append(np.sum(accumulated_psd[mask]))

    psd_sums = np.array(psd_sums, dtype=np.float32)
    psd_max = np.max(psd_sums)
    psd_min = np.min(psd_sums)
    if psd_max - psd_min > 0:
        psd_sums = (psd_sums - psd_min) / (psd_max - psd_min)
    
    psd_sums = psd_sums.tolist()

    # (C) 2次多項式フィット A, B, C
    xvals = np.array(R_VALUES, dtype=np.float32)
    yvals = np.array(psd_sums, dtype=np.float32)
    X_fit = np.vstack((xvals**2, xvals)).T
    linreg = LinearRegression().fit(X_fit, yvals)
    polyA = linreg.coef_[0]
    polyB = linreg.coef_[1]
    polyC = linreg.intercept_

    # (D) ROI輝度統計
    max_val = float(np.max(roi_float))
    min_val = float(np.min(roi_float))
    std_val = float(np.std(roi_float))
    bright_diff = max_val - min_val

    # 出力特徴: [ガボール周波数レスポンス] + [PSDサマリ] + [A,B,C] + [bright, std]
    features = (
        list(gabor_responses)
        + list(psd_sums)
        + [polyA, polyB, polyC, bright_diff, std_val]
    )
    print(f"特徴量の要素数: {len(features)}")
    return features

##############################################################################
# (3) 個体のAge & carrying capacity を導入
##############################################################################
def initialize_population(size):
    population = []
    existing_list = []
    for _ in range(size):
        point = pick_random_grid_point(IMAGE_WIDTH, IMAGE_HEIGHT, ROI_SIZE, existing_list)
        x_cand, y_cand = point
        ind = {
            "x": x_cand,
            "y": y_cand,
            "fitness": 0.0,
            "age": 0,
            "max_age": MAX_GENERATIONS,  # 適当に全員同じでOK
        }
        population.append(ind)
    return population

def adjust_carrying_capacity(current_capacity, survived_ratio):
    """
    生存率に応じてcarrying capacityを増減する簡単な例
    survived_ratio = (選択された個体数 / 現世代の個体数)
    """
    if survived_ratio < 0.2 and current_capacity > MIN_CAPACITY:
        return current_capacity - 10
    elif survived_ratio > 0.6 and current_capacity < MAX_CAPACITY:
        return current_capacity + 10
    print(f"  Survived ratio: {survived_ratio:.2f}, Current carrying capacity: {current_capacity}")
    return current_capacity

def select_population(population, carrying_capacity):
    # 1) fitness による選択
    selected = [ind for ind in population if ind["fitness"] >= FITNESS_THRESHOLD]
    # 2) age で除外
    selected = [ind for ind in selected if ind["age"] < ind["max_age"]]
    # carrying_capacity で削る
    # fit順にソートして上位だけ残す例
    selected.sort(key=lambda i: i["fitness"], reverse=True)
    return selected[:carrying_capacity]

def merge_close_clusters(population, labels, distance_threshold=50):
    """
    DBSCAN 等で得られたラベル配列 (labels) をもとに、
    各クラスタの重心が一定距離未満ならマージして新たなラベルを振り直す。

    Args:
        population (list of dict): 個体情報 ({"x":, "y":, ...} を含む)
        labels (np.ndarray): 各個体に対応するクラスタラベル (DBSCAN の出力)
        distance_threshold (float): 重心がこれ未満の距離なら同一クラスタにマージする

    Returns:
        np.ndarray: マージ後の新しいラベル配列
    """
    if len(population) == 0 or len(labels) == 0:
        return labels

    # -1 (外れ値) はクラスタとして扱わない
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    unique_labels = sorted(list(unique_labels))

    if not unique_labels:
        # 実質クラスタが無い場合はそのまま
        return labels

    # 各クラスタの重心を計算
    centroids = {}
    for lab in unique_labels:
        coords = np.array([[p["x"], p["y"]] 
                           for p,lab_ in zip(population, labels) if lab_ == lab])
        centroid = coords.mean(axis=0)
        centroids[lab] = centroid

    # 重心が近いクラスタ同士をまとめるために Union-Find or BFS で連結成分を探す
    adjacency = {lab: [] for lab in unique_labels}

    def euclid_dist(a, b):
        return np.linalg.norm(a - b)

    # 隣接リストを作る
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

    # 近いクラスタ同士を同じラベルに再割当
    for lab in unique_labels:
        if lab not in visited:
            group = bfs(lab)  # lab と近いクラスタを1つのグループにする
            for g in group:
                new_label_map[g] = merged_label_id
            merged_label_id += 1

    # 個体ごとの旧ラベルを新ラベルに置き換える
    new_labels = []
    for lab in labels:
        if lab == -1:
            new_labels.append(-1)
        else:
            new_labels.append(new_label_map[lab])
    return np.array(new_labels)


def reproduce_population(selected_population, total_size, candidate_points, roi_size, ratio_parent=0.7):
    """
    ratio_parent: 親の近くにオフスプリングを配置する割合 (0.0 ~ 1.0)
    """
    new_population = []
    n_offsprings = total_size - len(selected_population)

    existing_list = new_population + selected_population

    if len(selected_population) == 0:
        for _ in range(n_offsprings):
            point = pick_random_grid_point(IMAGE_WIDTH, IMAGE_HEIGHT, roi_size, existing_list)
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

    # 生成する比率を計算し、親個体ベースとランダムベースに分割
    n_parent_based = int(n_offsprings * ratio_parent)
    n_random = n_offsprings - n_parent_based

    possible_offsets = 70

    if n_parent_based > len(selected_population):
        parent_list = [np.random.choice(selected_population) for _ in range(n_parent_based)]
    else:
        parent_list = random.sample(selected_population, n_parent_based)
    
# --- (1) 親個体の「近く」にオフスプリングを配置 ---
    for parent in parent_list:
        point = pick_random_grid_near_parent(
            px=parent["x"],
            py=parent["y"],
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            roi_size=roi_size,
            existing_individuals=existing_list,
            offset_range=OFFSET_RANGE,  # ±何ピクセル探すか
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

    # --- (2) 残り n_random 個は完全ランダムにオフスプリングを配置 ---
    for _ in range(n_random):
        point = pick_random_grid_point(IMAGE_WIDTH, IMAGE_HEIGHT, roi_size,
                                       existing_list, max_tries=30)
        if point is None:
            break
        x_cand, y_cand = point
        offspring = {
            "x": x_cand,
            "y": y_cand,
            "fitness": 0.0,
            "age": 0,
            "max_age":  np.random.randint(5, MAX_GENERATIONS),
        }
        new_population.append(offspring)
        existing_list.append(offspring)
        
    # 親たちは ageを+1
    for ind in selected_population:
        ind["age"] += 1
    
    # 親世代を new_population に追加
    new_population.extend(selected_population)
    return new_population


##############################################################################
# (4) メインEAフロー: carrying_capacity を可変化 & PSD (5回累積 + Hanning + 周波数Gabor)
##############################################################################
def process_images(folder, trained_model_path, selector):
    from PIL import Image
    # 画像ファイルを取得する簡単な関数
    def get_image_files(fld):
        return sorted([os.path.join(fld,f) for f in os.listdir(fld) if f.endswith((".png", ".jpg"))])

    image_files = get_image_files(folder)
    print(f"Found {len(image_files)} images")

    # 学習済みモデル読込
    loaded_model = tf.keras.models.load_model(
        trained_model_path
    )

    # 周波数ドメインのガボールフィルタ生成 (例)
    freq_gabor_filters = create_freq_gabor_filter_bank(
        roi_size=ROI_SIZE,
        radii=GABOR_R,
        angles=GABOR_THETA
    )

    # 初期個体群
    candidate_points = generate_grid_points(IMAGE_WIDTH, IMAGE_HEIGHT, ROI_SIZE, spacing_x=X_SPACING, spacing_y=Y_SPACING)
    carrying_capacity = INITIAL_POPULATION_SIZE
    population = initialize_population(INITIAL_POPULATION_SIZE)

    for idx, img_path in enumerate(image_files):
        print(f"[Frame {idx+1}] Processing image: {img_path}")
        img_gray = Image.open(img_path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.array(img_gray, dtype=np.uint8)

        # 個体の適応度を計算
        evaluate_population(population, img_array, freq_gabor_filters, loaded_model, selector)

        # DBSCANでクラスタリング可視化等（省略可）
        labels = cluster_population(population, eps=EPS, min_samples=MIN_CLUSTER_SIZE)
        visualize_clusters_separately(population, labels, img_array,
                              ROI_SIZE=ROI_SIZE,
                              title=f"Clusters_{idx+1}")
        visualize_clusters(population, labels, img_array, f"Clusters {idx+1}")

        # 選択 + carrying capacity 調整
        survived = select_population(population, carrying_capacity)
        ratio = len(survived)/float(len(population)) if len(population)>0 else 0
        carrying_capacity = adjust_carrying_capacity(carrying_capacity, ratio)
        print(f"  Survived={len(survived)}, Next carrying_capacity={carrying_capacity}")

        # 生き残りが0なら打ち切り
        if len(survived) == 0:
            print("  No individuals selected, stopping.")
            # break

        # 繁殖
        population = reproduce_population(survived, carrying_capacity, candidate_points, ROI_SIZE)

    return population

##############################################################################
# (5) 個体ごとのfitness計算: PSDを5回累積 & モデルに入力
##############################################################################
def evaluate_population(population, image, freq_gabor_filters, loaded_model, selector):
    for individual in population:
        x, y = individual["x"], individual["y"]
        roi = image[y:y+ROI_SIZE, x:x+ROI_SIZE]

        # → PSD(5回累積)+Hanning+周波数ドメインガボール
        features = calculate_psd_features(
            roi, freq_gabor_filters, num_accumulate=5
        )

        # モデルに入力 (reshape等)
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        features_array_60 = selector.transform(features_array)
        # 推定値(=fitness) を格納
        fitness_val = loaded_model.predict(features_array_60)[0,0]
        individual["fitness"] = float(fitness_val)

##############################################################################
# (6) DBSCANなどのクラスタリング例 (必要に応じて)
##############################################################################
def cluster_population(population, eps=60, min_samples=5):
    if not population:
        return []
    coords = np.array([[ind["x"], ind["y"]] for ind in population])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = merge_close_clusters(population, db.labels_, distance_threshold=DISTANCE_THRESHOLD)
    
    return labels

def visualize_clusters(population, labels, img_array, title="Clusters"):
    import matplotlib.pyplot as plt
    import os

    out_dir = "/home/yoshi-22/Bench2Drive/reliability/out_det"
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    unique_labels = set(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))
    cluster_area_texts = []
    

    for idx, k in enumerate(unique_labels):
        class_member_mask = (labels == k)

        roi_corners = []
        for ind, is_member in zip(population, class_member_mask):
            if not is_member:
                continue
            x0, y0 = ind["x"], ind["y"]
            corners = [
                (x0, y0),
                (x0 + ROI_SIZE, y0),
                (x0 + ROI_SIZE, y0 + ROI_SIZE),
                (x0, y0 + ROI_SIZE)
            ]
            roi_corners.extend(corners)
        if len(roi_corners) == 0:
            # このクラスタに点が無いならスキップ
            continue

        xy = np.array(np.array(roi_corners))
        xy2 = np.array([[p["x"], p["y"]] for p,m in zip(population,class_member_mask) if m])
        if k == -1:
            plt.scatter(xy2[:,0], xy2[:,1], c="k", marker="x", label=None)
        else:
            color_val = colors(idx)
            plt.scatter(xy2[:,0], xy2[:,1], marker="o", color=color_val, label=None)

            if len(xy) > 3:
                hull = ConvexHull(xy)
                area = hull.area
                hull_vertices = hull.vertices
                hull_points = xy[hull_vertices]

                print(f"  Cluster {k}: {len(xy)} points, area={area:.2f}")
                plt.fill(hull_points[:,0], hull_points[:,1], color=color_val, alpha=0.3)
                cluster_area_texts.append(f"C{k}={area:.2f}")
    
    if cluster_area_texts:
        area_text = ', '.join(cluster_area_texts)
        print(f"  Cluster areas: {area_text}")
        plt.title(f"{title} ({area_text})")
    else:    
        plt.title(title)

    plt.imshow(img_array, cmap="gray", alpha=0.5)
    plt.legend()
    # plt.show()
    save_path = os.path.join(out_dir, f"{title}.png")
    plt.savefig(save_path)
    plt.close()

# CNNモデル構築
def build_cnn_model():
    # model = Sequential([
    #     Dense(60, activation="relu", input_shape=(200,)),
    #     Dense(60, activation="relu"),
    #     Dense(60, activation="relu"),
    #     Dense(1, activation="sigmoid")
    # ])
    model = Sequential([
        Dense(128, activation="relu", input_shape=(60,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# 画像ペアを処理する関数
def process_image(args):
    img_array, label_array, filters, roi_size, n_rois = args
    X = []
    Y = []
    for _ in range(n_rois):
        max_tries = 10
        # ランダムなROIの位置を決定
        for i in range(max_tries):
            x = np.random.randint(0, img_array.shape[1] - roi_size)
            y = np.random.randint(0, img_array.shape[0] - roi_size)
            roi = img_array[y:y+roi_size, x:x+roi_size]
            
            # ROIに基づいて特徴量を計算
            features = calculate_psd_features(roi, filters, num_accumulate=5)
            
            # ROI内のラベル値の平均を計算（正解データとして利用）
            roi_label_values = label_array[y:y+roi_size, x:x+roi_size]
            average_label = np.mean(roi_label_values / 255.0)
            if average_label >= 0.5:
                average_label = 1.0
            else:
                average_label = 0.0
            
            if average_label == 0.0 and i != max_tries - 1:
                # ラベルが0のものはスキップ
                continue
            else:
                X.append(features)
                Y.append(average_label)
    
    return X, Y

# 画像とラベルを事前に読み込む
def load_image(img_path, label_path):
    img = Image.open(img_path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    label_img = Image.open(label_path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    return np.array(img), np.array(label_img)

# 並列処理を使用したデータ生成関数
def generate_training_data_parallel(image_paths, label_paths, filters, roi_size=49, n_rois=50):
    """
    並列処理を使用してトレーニングデータを生成する関数。

    Args:
        image_paths (list): 入力画像のパスのリスト。
        label_paths (list): ラベル画像のパスのリスト。
        filters (list): ガボールフィルタのリスト。
        roi_size (int): ROIのサイズ（デフォルトは49）。
        n_rois (int): 各画像から生成するROIの数（デフォルトは50）。

    Returns:
        tuple: 特徴量の配列 (X) とラベルの配列 (Y)。
    """
    
    # すべての画像をロード
    with Pool(processes=cpu_count()) as pool:
        loaded_data = pool.starmap(load_image, zip(image_paths, label_paths))
    
    # プロセスプールの準備
    args = [
        (img_array, label_array, filters, roi_size, n_rois)
        for img_array, label_array in loaded_data
    ]
    
    # 並列処理でデータ生成
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, args)
    
    # 結果を統合
    X = []
    Y = []
    for res_X, res_Y in results:
        X.extend(res_X)
        Y.extend(res_Y)
    
    return np.array(X), np.array(Y)


def anova_feature_selection(X, y, n_features=60):
    """
    ANOVA F値に基づいて特徴量を選択する関数。

    Args:
        X (np.ndarray): 入力特徴量の配列。
        y (np.ndarray): 出力ラベルの配列。
        n_features (int): 選択する特徴量の数。

    Returns:
        np.ndarray: 選択された特徴量の配列。
    """
    selector = SelectKBest(f_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    print(f"Feature selection: {X_new.shape[1]} features selected.")
    # （もしも特徴量名一覧があるなら、同じ順序のリストとして渡す）
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    # 可視化
    visualize_selected_features(selector, feature_names=feature_names)
    return X_new, selector

# 学習と評価用コード
def train_and_evaluate_model(X_train, y_train, X_test, y_test, use_anova=True, n_features=60):

    selector = None
    if use_anova:
        X_train, selector = anova_feature_selection(X_train, y_train, n_features=n_features)
        X_test = selector.transform(X_test)

    model = build_cnn_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler]
    )

    # テストデータに対する予測
    predictions = model.predict(X_test).flatten()
    print(f"最小予測値: {predictions.min()}, 最大予測値: {predictions.max()}")
    pred_labels = (predictions >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred_labels)
    prec = precision_score(y_test, pred_labels, zero_division=0)
    rec = recall_score(y_test, pred_labels, zero_division=0)
    f1 = f1_score(y_test, pred_labels, zero_division=0)
    print("=== Classification Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")


    # 混同行列や Classification Report
    cm = confusion_matrix(y_test, pred_labels)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_test, pred_labels))

    # もし可視化したければ確率 vs ラベルの散布図など
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, 'ro', label="Predicted Probability")
    plt.plot(y_test, 'bo', label="True Label")
    plt.title("Predicted Probability vs True Label")
    plt.xlabel("Sample Index")
    plt.ylabel("Label / Probability")
    plt.legend()
    plt.show()

    return model, selector

# categories = ["front", "front_left", "front_right", "back", "back_left", "back_right"]

# image_paths = []
# label_paths = []

# for category in categories:
#     for i in range(1, 4):
#         image_pattern = os.path.join("./ri_image", f"{category}_{i}_*.png")
#         label_pattern = os.path.join("./ri_label", f"{category}_{i}_*.png")

#         image_paths += sorted(glob.glob(image_pattern))
#         label_paths += sorted(glob.glob(label_pattern))

# print(f"Number of images: {len(image_paths)}")
# print(f"Number of labels: {len(label_paths)}")
# # 画像とラベルの数が一致していることを確認
# if len(image_paths) != len(label_paths):
#     raise ValueError("画像とラベルの数が一致しません。")

# # 各ペアのファイル名が一致していることを確認
# for img_path, lbl_path in zip(image_paths, label_paths):
#     img_name = os.path.basename(img_path)
#     lbl_name = os.path.basename(lbl_path)
#     if img_name != lbl_name:
#         raise ValueError(f"画像とラベルが一致していません: {img_name} vs {lbl_name}")

# print("すべての画像とラベルが正しくマッチしています。")

# filters = create_freq_gabor_filter_bank(roi_size=ROI_SIZE, radii=GABOR_R, angles=GABOR_THETA)
# X, y = generate_training_data_parallel(image_paths, label_paths, filters, roi_size=ROI_SIZE, n_rois=50)

# # ラベルの分布をプロット
# plot_label_distribution(y)

# # 入力画像とラベル画像が存在するか確認
# for img_path in image_paths + label_paths:
#     if not os.path.exists(img_path):
#         raise FileNotFoundError(f"File not found: {img_path}")

# # データ分割（仮）
# X_train, X_test = X[:len(X)*9//10], X[len(X)*9//10:]
# y_train, y_test = y[:len(y)*9//10], y[len(y)*9//10:]

# # 学習と評価
# model, selector = train_and_evaluate_model(X_train, y_train, X_test, y_test)
# model_save_path = "./model_small.h5"
# model.save(model_save_path)
# joblib.dump(selector, "anova_selector_small.pkl")
# print(f"Model saved to {model_save_path}")


def main():
    trained_model_path = "./model.h5"  # 学習ずみモデルのパス
    selector = joblib.load("anova_selector.pkl")
    image_folder = "/home/yoshi-22/Bench2Drive/output_cluster_left/output_images/front"        # クラスタリングを行いたい画像フォルダ
    # image_folder = "/home/yoshi-22/UniAD/data/nuscenes/samples/CAM_FRONT"        # クラスタリングを行いたい画像フォルダ

    # クラスタリング可視化を実行
    population = process_images(image_folder, trained_model_path, selector)

    # population には最終世代の個体情報が格納されています
    # 追加で何か分析したい場合はここで行う
    
if __name__ == "__main__":
    main()
