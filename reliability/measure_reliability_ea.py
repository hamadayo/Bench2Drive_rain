import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIL import Image
from scipy.fftpack import fft2
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import MeanSquaredError
import cv2
from sklearn.cluster import DBSCAN
import random

# 定数定義
IMAGE_WIDTH, IMAGE_HEIGHT = 900, 512  # 画像サイズ
ROI_SIZE = 49  # ROI (Region of Interest) のサイズ
INITIAL_POPULATION_SIZE = 50  # 初期個体数
IMAGE_FOLDER = "/home/yoshi-22/Bench2Drive/eval_v1/RouteScenario_1711_rep0_Town12_ParkingCutIn_1_15_12_18_16_56_22_rain_soso/rgb_front"  # 画像フォルダ
MAX_GENERATIONS = 5  # 最大世代数
FITNESS_THRESHOLD = 0.2  # 適応度閾値
R_VALUES = range(5, 25, 4)  # 半径範囲
GABOR_R = [3, 6, 12, 18, 21]
GABOR_THETA = [i for i in range(0, 180, 5)]  # ガボール方向

def get_image_files(folder):
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    )

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

    # 上半分のみを選択する場合
    if upper_half_only:
        mask &= (y <= center_y)

    return mask


# ガボールフィルタ生成
def gabor_filter_bank():
    filters = []
    for r in GABOR_R:
        for theta in GABOR_THETA:
            sigma = r / 2
            lambd = r
            gamma = 0.5
            psi = 0
            kernel = cv2.getGaborKernel((ROI_SIZE, ROI_SIZE), sigma, np.deg2rad(theta), lambd, gamma, psi, ktype=cv2.CV_32F)
            filters.append(kernel)
    return filters

# ガボールフィルタ適用
def apply_gabor_filters(roi, filters):
    responses = []
    for kernel in filters:
        response = cv2.filter2D(roi, cv2.CV_32F, kernel)
        responses.append(np.sum(response))
    return responses

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

    # 上半分のみを選択する場合
    if upper_half_only:
        mask &= (y <= center_y)

    return mask

# PSD累積特徴と多項式フィッティング
def calculate_psd_features(roi):
    gray_roi = roi.astype(np.float32)
    gray_roi = np.log1p(gray_roi)
    filtered_roi = gaussian_filter(gray_roi, sigma=1)
    fft_roi = fft2(filtered_roi)
    psd = np.abs(fft_roi)**2

    # 半径ごとのPSD合計
    psd_accumulated = []
    for r in R_VALUES:
        mask = create_circular_mask(ROI_SIZE, ROI_SIZE, r, upper_half_only=True)
        psd_accumulated.append(np.sum(psd[mask]))

    # 2次多項式フィッティング
    x = np.array(R_VALUES)
    y = np.array(psd_accumulated)

    X = np.vstack((x**2, x)).T
    
    model = LinearRegression().fit(X, y)
    A = model.coef_[0]  # x の係数
    B = model.coef_[1]  # x^2 の係数
    C = model.intercept_  # 切片
    
    return psd_accumulated, A, B, C

# ROI輝度の特徴計算
def calculate_brightness_features(roi):
    max_val = np.max(roi)
    min_val = np.min(roi)
    std_dev = np.std(roi)
    return max_val - min_val, std_dev

# 適応度計算
def calculate_fitness(x, y, image, filters):
    roi = image[y:y+ROI_SIZE, x:x+ROI_SIZE]

    gabor_features = apply_gabor_filters(roi, filters)
    psd_features, A, B, C = calculate_psd_features(roi)
    brightness_diff, brightness_std = calculate_brightness_features(roi)

    # featuresは配列
    features = gabor_features + psd_features + [A, B, C, brightness_diff, brightness_std]
    # モデルに入力するための前処理
    features_array = np.array(features).reshape(1, -1)  # モデル入力の形状に変換

    # モデルによる適応度の計算
    fitness = trained_model.predict(features_array)[0, 0]  # モデルの出力（1つのスカラー値）
    return fitness

def preprocess_roi(roi):
    gray_roi = roi.astype(np.float32)
    #　対数変換
    gray_roi = np.log1p(gray_roi)
    # ガウシアンフィルタ
    filtered_roi = gaussian_filter(gray_roi, sigma=1)

    # フーリエ変換とPSDの計算
    fft_roi = fft2(filtered_roi)
    psd_roi = np.abs(fft_roi)**2
    # psdの値を２次元配列でそのまま返す
    return psd_roi

# 仮の適応度計算 (実際はニューラルネットワークで計算)
# def calculate_fitness(x, y, image):
#     roi = image[y:y+ROI_SIZE, x:x+ROI_SIZE]

#     # psdを５回累積
#     accumulated_psd = np.zeros((ROI_SIZE, ROI_SIZE))
#     for _ in range(5):
#         psd_roi = preprocess_roi(roi)
#         accumulated_psd += psd_roi

#     print(f"  Fitness: {np.sum(accumulated_psd)}")
#     # 高周波成分の総和を適応度とする
#     # high_freq_sum = np.sum(accumulated_psd[3:, 3:])
#     high_freq_sum = np.sum(accumulated_psd[ROI_SIZE//2:])
#     print(f"  High frequency sum: {high_freq_sum}, {np.max(accumulated_psd)}, {high_freq_sum / np.max(accumulated_psd)}")
#     return high_freq_sum / np.max(accumulated_psd)

# DBSCANを用いたクラスタリング
def cluster_population(population, eps=50, min_samples=5):
    """
    DBSCANを用いて個体群をクラスタリングする関数。

    Args:
        population (list): 個体群のリスト。
        eps (float): DBSCANのεパラメータ（近傍の半径）。
        min_samples (int): DBSCANのmin_samplesパラメータ（クラスター形成に必要な最小点数）。

    Returns:
        list: クラスタリング結果（各個体に対するクラスタラベル）。
    """
    if not population:
        return []

    # 個体の座標を抽出
    coordinates = np.array([[ind["x"], ind["y"]] for ind in population])

    # DBSCANの適用
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = db.labels_

    return labels

def calculate_cluster_coverage(population, labels, image_shape, roi_size=49):
    """
    クラスタリングで覆われた面積の割合を計算する関数。

    Args:
        population (list): 個体群のリスト。
        labels (list): クラスタラベルのリスト。
        image_shape (tuple): 画像の形状 (height, width)。
        roi_size (int): ROIのサイズ（デフォルトは49）。

    Returns:
        float: クラスタリングで覆われた面積の割合（0から1）。
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for ind, label in zip(population, labels):
        if label != -1:  # クラスタに属する個体のみ
            x, y = ind["x"], ind["y"]
            # ROIの範囲をマスクに設定
            mask[y:y+roi_size, x:x+roi_size] = 1

    covered_pixels = np.sum(mask)
    total_pixels = height * width
    coverage_ratio = covered_pixels / total_pixels

    return coverage_ratio

# クラスタリング結果の可視化
def visualize_clusters(population, labels, img, title):
    """
    クラスタリング結果を可視化する関数。

    Args:
        population (list): 個体群のリスト。
        labels (list): クラスタラベルのリスト。
        img (numpy.ndarray): 元の画像データ。
        title (str): プロットのタイトル。
    """
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # ノイズ
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = np.array([[ind["x"], ind["y"]] for ind, flag in zip(population, class_member_mask) if flag])
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker='o', label=f'Cluster {k}' if k != -1 else 'Noise')

    plt.imshow(img, cmap="gray", interpolation="nearest", alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.show()


# 初期個体群を生成
def initialize_population(size):
    return [
        {"x": np.random.randint(0, IMAGE_WIDTH - ROI_SIZE),
         "y": np.random.randint(0, IMAGE_HEIGHT - ROI_SIZE),
         "fitness": 0,
         "age": 0} for _ in range(size)
    ]

# 適応度計算
def evaluate_population(population, image, filters):
    for individual in population:
        individual["fitness"] = calculate_fitness(individual["x"], individual["y"], image, filters)

# 適応度に基づく選択
def select_population(population):
    return [ind for ind in population if ind["fitness"] >= FITNESS_THRESHOLD and ind["age"] < MAX_GENERATIONS]
    

# 繁殖 (交叉と突然変異)
def reproduce_population(selected_population, total_size):
    new_population = []
    while len(new_population) < total_size - len(selected_population):
        parent = np.random.choice(selected_population)
        offspring = {
            "x": np.clip(parent["x"] + np.random.randint(-ROI_SIZE, ROI_SIZE), 0, IMAGE_WIDTH - ROI_SIZE),
            "y": np.clip(parent["y"] + np.random.randint(-ROI_SIZE, ROI_SIZE), 0, IMAGE_HEIGHT - ROI_SIZE),
            "fitness": 0,
            "age": 0
        }
        new_population.append(offspring)
        # print(f"  Offspring: {offspring['x']}, {offspring['y']}")
    for individual in selected_population:
        individual["age"] += 1
        # print(f"  Parent: {individual['x']}, {individual['y']}, age: {individual['age']}")
    new_population.extend(selected_population)
    return new_population

# メインアルゴリズム
def process_images(folder, filters):
    image_files = get_image_files(folder)
    print(f"Found {len(image_files)} images")

    population = initialize_population(INITIAL_POPULATION_SIZE)
    for idx, image_path in enumerate(image_files):
        print(f"Processing image {idx + 1}/{len(image_files)}: {image_path}")

        img = Image.open(image_path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.array(img)
        # 適応度を計算
        evaluate_population(population, img_array, filters)

        labels = cluster_population(population, eps=50, min_samples=5)
        visualize_clusters(population, labels, img_array, f"Clusters for {os.path.basename(image_path)}")

        visualize_population(population, img_array, f"Result for {os.path.basename(image_path)}")

        # 選択
        selected_population = select_population(population)
        print(f"  Selected {len(selected_population)} individuals")

        if len(selected_population) == 0:
            print("  No individuals selected, stopping.")
            break
        # 繁殖
        population = reproduce_population(selected_population, INITIAL_POPULATION_SIZE)

    return population

# 結果の可視化
def visualize_population(population, img, title):
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for individual in population:
        x, y = individual["x"], individual["y"]
        image[y:y+ROI_SIZE, x:x+ROI_SIZE] += 1
    plt.imshow(img, cmap="gray", interpolation="nearest", alpha=0.7)
    plt.imshow(image, cmap="hot", interpolation="nearest", alpha=0.5)
    plt.colorbar(label="ROI Density")
    plt.title(title)
    plt.show()

filters = gabor_filter_bank()
model_path = "./model.h5"  # 学習済みモデルのパス
trained_model = tf.keras.models.load_model(
    model_path,
    custom_objects={"mse": MeanSquaredError()}
)
# 実行
process_images(IMAGE_FOLDER, filters)
