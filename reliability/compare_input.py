import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIL import Image
from scipy.fftpack import fft2
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import cv2
import glob
import random
from multiprocessing import Pool, cpu_count
from sklearn.feature_selection import f_classif

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
R_VALUES = range(5, 25, 4)  # 半径範囲
GABOR_R = [3, 6, 12, 18, 21]
GABOR_THETA = [i for i in range(0, 180, 5)]  # ガボール方向

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

            # 中心周波数 fc = r (例)
            # 簡易的に "fc付近をガウシアンで強調" するモデル
            # dist^2 = (sqrt(U^2 + V^2) - r)^2

            # ガウシアン
            gb = np.exp(-0.5 * ((Ur)**2 + (Vr - r)**2) / sigma**2) * np.cos(2 * np.pi * r * Ur / roi_size)

            gb /= np.linalg.norm(gb)

            # 位相シフト等が必要なら追加で掛け合わせる
            # ここでは振動成分を省いて "輪状ガウシアン" としている
            # 実際のガボールは cos( ... ) などを掛けることが多い
            filters.append(gb)
    
    return filters

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
    cx = ROI_SIZE // 2
    cy = ROI_SIZE // 2
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
    return features

def create_circular_mask(width, height, radius, upper_half_only=False):
    """
    円形マスクまたは半円マスクを作成する関数。
    """
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (distance_from_center >= radius - 0.5) & (distance_from_center < radius + 0.5)
    mask &= (y <= center_y )
    mask &= (x <= center_x )
    return mask

# CNNモデル構築
def build_cnn_model():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(190,)),
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
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# 画像ペアを処理する関数
def process_image(args):
    img_array, label_array, filters, roi_size, n_rois = args
    X = []
    Y = []
    for _ in range(n_rois):
        # ランダムなROIの位置を決定
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


# 学習と評価用コード
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
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
    predictions_original = np.clip(predictions, a_min=0, a_max=1)

    # MSEの計算
    mse = mean_squared_error(y_test, predictions_original)
    r2 = r2_score(y_test, predictions_original)
    print(f"MSE on test data: {mse}")
    print(f"R^2 Score on test data: {r2}")

    # 結果の可視化
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color="blue", label="True Labels")
    plt.scatter(range(len(predictions_original)), predictions_original, color="red", label="Predictions")
    plt.legend()
    plt.title("True Labels vs Predictions")
    plt.show()

    return model


if __name__ == "__main__":
    # ======== 1) Gaborフィルタ生成 ========
    filters = create_freq_gabor_filter_bank(roi_size=ROI_SIZE, radii=GABOR_R, angles=GABOR_THETA)

    print(f"Number of Gabor filters: {type(filters)}")

    # *****************************************************************
    # ここからが【検証用】追加コード。
    # a.png（雨滴まみれ）と b.png（雨滴なし）を用いて、
    # 特徴量の違いを可視化 (print) する処理を追加した例。
    # *****************************************************************
    
    # (a) a.png と b.png を読み込み（グレースケール＆リサイズ）
    #    ※ パスはご自身の環境に合わせて書き換えてください
    test_img_a_path = "/home/yoshi-22/Bench2Drive/reliability/compare/a.png"
    test_img_b_path = "/home/yoshi-22/Bench2Drive/reliability/compare/b.png"
    
    img_a = Image.open(test_img_a_path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_b = Image.open(test_img_b_path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    # NumPy配列化
    img_a_array = np.array(img_a)
    img_b_array = np.array(img_b)

    # (b) 適当にROIの左上座標 (roi_x, roi_y) を決める
    #     ここでは「画像の中心 - ROI_SIZE/2」を左上とするイメージ。
    x = (IMAGE_WIDTH // 2) - (ROI_SIZE // 2)
    y = (IMAGE_HEIGHT // 2) - (ROI_SIZE // 2)

    roi_a = img_a_array[y:y+ROI_SIZE, x:x+ROI_SIZE]
    roi_b = img_b_array[y:y+ROI_SIZE, x:x+ROI_SIZE]


    # (c) それぞれの画像からROIを切り出して特徴量を計算
    features_a = calculate_psd_features(roi_a, filters)
    features_b = calculate_psd_features(roi_b, filters)

    # 特徴量ベクトルのサイズなどを把握したい場合
    print("====================================================")
    print(f"[Debug] ROIサイズ: {ROI_SIZE}x{ROI_SIZE}")
    print(f"[Debug] Gaborフィルタ数: {len(filters)}")
    print(f"[Debug] PSD半径R_VALUES: {list(R_VALUES)}")

    # (d) 結果をprint
    #    gaborフィルタの応答(合計) + psd_features(累積) + [A, B, C, brightness_diff, brightness_std]
    print("====================================================")
    print("[a.png] 特徴量 (要素数:", len(features_a), ")")
    print(features_a)
    print("====================================================")
    print("[b.png] 特徴量 (要素数:", len(features_b), ")")
    print(features_b)
    print("====================================================")

    # *****************************************************************
    # ここまでが検証用コード。ROIが同じ位置なら
    # a.png と b.png で特徴量がどう変わるか比較しやすいはずです。
    # *****************************************************************

    # ----------------------------------------------------------------
    # 以下は「train_and_evaluate_model」を使った通常の学習コード例
    # （実際に a.png, b.png などをたくさん用意して学習する場合）
    # ----------------------------------------------------------------

    categories = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    # categories = ["back"]  # ←例としてカテゴリ"back"のみ

    image_paths = []
    label_paths = []

    for i in range (0, 208):
        for category in categories:
            image_pattern = os.path.join("./ri_image", f"raindrop_{i}_{category}_*.png")
            label_pattern = os.path.join("./ri_label", f"raindrop_{i}_{category}_*.png")

            image_paths += sorted(glob.glob(image_pattern))
            label_paths += sorted(glob.glob(label_pattern))

    print(f"Number of images: {len(image_paths)}")
    print(f"Number of labels: {len(label_paths)}")
    # 画像とラベルの数が一致していることを確認
    if len(image_paths) != len(label_paths):
        raise ValueError("画像とラベルの数が一致しません。")

    # 各ペアのファイル名が一致していることを確認
    for img_path, lbl_path in zip(image_paths, label_paths):
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(lbl_path)
        if img_name != lbl_name:
            raise ValueError(f"画像とラベルが一致していません: {img_name} vs {lbl_name}")
    print("すべての画像とラベルが正しくマッチしています。")

    # (2) 特徴量生成
    X, y = generate_training_data_parallel(
        image_paths, label_paths, 
        filters, 
        roi_size=ROI_SIZE, 
        n_rois=50
    )

    # ラベルの分布をプロット
    plot_label_distribution(y)

    # 入力画像とラベル画像が存在するか確認
    for img_path in image_paths + label_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

    # (3) データ分割（仮）
    split_idx = len(X)*9//10
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # (4) 学習と評価
    model = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    model_save_path = "./model.h5"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
