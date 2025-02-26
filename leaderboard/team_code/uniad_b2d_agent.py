import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from Bench2DriveZoo.team_code.pid_controller import PIDController
from Bench2DriveZoo.team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.optimize import fsolve

import csv
from raindrop.dropgenerator import generateDrops
from raindrop.config import cfg
from PIL import Image
from raindrop.raindrop import Raindrop
import random
import joblib
from reliability.use_model_func import create_freq_gabor_filter_bank, generate_grid_points, initialize_population, ea_process_single_frame, visualize_clusters
import tensorflow as tf
from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

valid_names = [
    'img_backbone.layer4.2'
]

CAM_PATHS = {
    "CAM_FRONT": ("/home/yoshi-22/Bench2Drive/reliability/image_front", "/home/yoshi-22/Bench2Drive/reliability/label_front"),
    "CAM_FRONT_RIGHT": ("/home/yoshi-22/Bench2Drive/reliability/image_front_right", "/home/yoshi-22/Bench2Drive/reliability/label_front_right"),
    "CAM_FRONT_LEFT": ("/home/yoshi-22/Bench2Drive/reliability/image_front_left", "/home/yoshi-22/Bench2Drive/reliability/label_front_left"),
    "CAM_BACK": ("/home/yoshi-22/Bench2Drive/reliability/image_back", "/home/yoshi-22/Bench2Drive/reliability/label_back"),
    "CAM_BACK_LEFT": ("/home/yoshi-22/Bench2Drive/reliability/image_back_left", "/home/yoshi-22/Bench2Drive/reliability/label_back_left"),
    "CAM_BACK_RIGHT": ("/home/yoshi-22/Bench2Drive/reliability/image_back_right", "/home/yoshi-22/Bench2Drive/reliability/label_back_right"),
}

CAM_LOG_PATHS = {
    "CAM_FRONT":       "/home/yoshi-22/Bench2Drive/output_cluster/reliability/reliability_log_front.txt",
    "CAM_FRONT_LEFT":  "/home/yoshi-22/Bench2Drive/output_cluster/reliability/reliability_log_front_left.txt",
    "CAM_FRONT_RIGHT": "/home/yoshi-22/Bench2Drive/output_cluster/reliability/reliability_log_front_right.txt",
    "CAM_BACK":        "/home/yoshi-22/Bench2Drive/output_cluster/reliability/reliability_log_back.txt",
    "CAM_BACK_LEFT":   "/home/yoshi-22/Bench2Drive/output_cluster/reliability/reliability_log_back_left.txt",
    "CAM_BACK_RIGHT":  "/home/yoshi-22/Bench2Drive/output_cluster/reliability/reliability_log_back_right.txt"
}

HEATMAP_MEMORY_FRAMES = 2  # ヒートマップのメモリフレーム数

heatmap_hist = {
    "CAM_FRONT": deque(maxlen=HEATMAP_MEMORY_FRAMES),
    "CAM_FRONT_RIGHT": deque(maxlen=HEATMAP_MEMORY_FRAMES),
    "CAM_FRONT_LEFT": deque(maxlen=HEATMAP_MEMORY_FRAMES),
    "CAM_BACK": deque(maxlen=HEATMAP_MEMORY_FRAMES),
    "CAM_BACK_LEFT": deque(maxlen=HEATMAP_MEMORY_FRAMES),
    "CAM_BACK_RIGHT": deque(maxlen=HEATMAP_MEMORY_FRAMES),
}

activations = {}

IMAGE_WIDTH, IMAGE_HEIGHT = 900, 512   # 画像サイズ
ROI_SIZE = 49                         # ROI (Region of Interest) のサイズ
R_VALUES = list(range(1, 15, 1)) + list(range(18, 21, 1))  # 半径範囲
GABOR_R = [2, 3, 4, 5]
GABOR_THETA = [i for i in range(20, 90, 5)]

# big
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
# EPS = 60  # DBSCANの距離閾値
# MIN_CLUSTER_SIZE = 5  # DBSCANの最小クラスタサイズ

# small
MIN_CAPACITY = 20   # データ生成の最小容量
MAX_CAPACITY = 140  # データ生成の最大容量
MAX_GENERATIONS = 20  # 個体の最大世代数
FITNESS_THRESHOLD = 0.8
INITIAL_POPULATION_SIZE = 70
X_SPACING = 20 # グリッドの横方向間隔
Y_SPACING = 20  # グリッドの縦方向間隔
OFFSET_RANGE = 40  # 親近傍での候補点ずらし範囲
CONFLICT_THRESHOLD = 20  # 衝突判定の閾値
DISTANCE_THRESHOLD = 35  # クラスタ重心のマージ判定距離
EPS = 50  # DBSCANの距離閾値
MIN_CLUSTER_SIZE = 5


# Raindrop cacheを基にクラスタリング生成
def cluster_raindrops_dbscan(raindrops, eps=20.0, min_samples=1):
    """
    `raindrops`: Raindropオブジェクトのリスト (drop.center, drop.radius などを持つ)

    eps        : DBSCANの距離閾値。 (中心距離 - 半径の和) <= eps なら同クラスタとみなす
    min_samples: クラスタ最小サイズ

    戻り値: labels (各雨滴のクラスタIDを格納した配列)
    """
    if not raindrops:
        return np.array([])

    n = len(raindrops)
    dist_mat = np.zeros((n, n), dtype=np.float32)

    # 中心 & 半径を取り出す
    # ここでは (float(cx), float(cy)) と float(r) にキャストしておく
    centers_radii = []
    for drop in raindrops:
        cx, cy = drop.center
        r      = drop.radius
        centers_radii.append((float(cx), float(cy), float(r)))

    # カスタム距離行列
    for i in range(n):
        (x_i, y_i, r_i) = centers_radii[i]
        for j in range(i+1, n):
            (x_j, y_j, r_j) = centers_radii[j]
            dx = x_i - x_j
            dy = y_i - y_j
            center_dist = np.sqrt(dx*dx + dy*dy)
            # (中心間距離 - 半径の和) がマイナスなら 0 にクリップ
            d = max(0.0, center_dist - (r_i + r_j))
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    db.fit(dist_mat)
    labels = db.labels_
    return labels

def build_raindrop_mask_from_cache(raindrops, labels, out_h=512, out_w=900,
                                   orig_w=1600, orig_h=900):
    """
    RAINDROP_CACHEの雨滴たち + DBSCANのクラスタラベルを使って、
    512x900サイズの雨滴マスクを生成するサンプル。

    Args:
      raindrops: Raindropオブジェクトのリスト (drop.center=(x,y), drop.radius=r, ...)
      labels   : cluster_raindrops_dbscan() の結果
      out_h, out_w: 出力マスクのサイズ (例: 512 x 900)
      orig_w, orig_h: 雨滴の (x,y,r) が想定されているもとの画像サイズ (例: 1600 x 900)
                      → これを out_w, out_h にリサイズして描画

    Returns:
      raindrop_mask: shape=(out_h, out_w), dtype=uint8 (0 or 1)
    """
    mask = np.zeros((out_h, out_w), dtype=np.uint8)

    # スケーリング比
    scale_x = out_w / float(orig_w)
    scale_y = out_h / float(orig_h)

    for drop, lab in zip(raindrops, labels):
        if lab == -1:
            # DBSCAN外れ値扱いの場合は無視するなら continue
            # もしくは外れ値でもマスクに含めるなら何もしない
            continue

        (cx, cy) = drop.center
        r = drop.radius
        
        # (cx, cy, r) をリスケール
        # もしアスペクト比が違うなら本来はX, Y方向で別々にスケールするが
        # 円が潰れるかもしれないので、実用上は同じスケールにするか要検討
        # ここは簡易的に (scale_x) だけ適用する例
        cx_2d = int(cx * scale_x)
        cy_2d = int(cy * scale_y)
        # r_2d  = int(r  * scale_x)
        r_2d = r
        
        # 円を描画
        cv2.circle(mask, (cx_2d, cy_2d), r_2d, 1, thickness=-1)

    return mask

def build_raindrop_mask(population, labels, image_h, image_w, roi_size=49):
    """
    population: [{x, y, fitness, ...}, ...]
    labels: DBSCAN結果 (len=population数) -1はノイズ
    戻り値: raindrop_mask: shape=(image_h, image_w), dtype=np.uint8 (0 or 1)
    """
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    if len(population) == 0 or len(labels) == 0:
        return mask
    
    for (indiv, lab) in zip(population, labels):
        # クラスタラベルが -1 は除外（ノイズ）
        if lab == -1:
            continue
        x0, y0 = indiv["x"], indiv["y"]
        # ROI領域を1に
        x1 = min(x0 + roi_size, image_w)
        y1 = min(y0 + roi_size, image_h)
        mask[y0:y1, x0:x1] = 1
    
    return mask

def tensor_to_grayscale_map(feature_map: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """
    feature_map: shape=(C, H', W'), torch.Tensor
    out_h, out_w: 出力する画像サイズ
    戻り値: shape=(out_h, out_w), 0〜1 に正規化されたfloat32配列
    """
    # 1) チャネル平均 -> (H', W')
    heatmap_2d = feature_map.mean(dim=0)  # shape=(H', W')

    # 2) numpyへ移行 & min-max正規化
    heatmap_2d = heatmap_2d.cpu().numpy()
    hmin, hmax = heatmap_2d.min(), heatmap_2d.max()
    if hmax - hmin > 1e-9:
        heatmap_2d = (heatmap_2d - hmin) / (hmax - hmin)
    else:
        heatmap_2d[:] = 0.0
    # float32へ
    heatmap_2d = heatmap_2d.astype(np.float32)

    # 3) リサイズ (cv2.INTER_LINEARなどで拡大)
    heatmap_resized = cv2.resize(heatmap_2d, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return heatmap_resized

def compute_reliability(heatmap: np.ndarray, raindrop_mask: np.ndarray) -> float:
    """
    heatmap:       shape=(H, W), 0～1のfloat32配列
    raindrop_mask: shape=(H, W), 0 or 1のuint8/float配列
    """
    heatmap_sum = np.sum(heatmap)
    if heatmap_sum < 1e-9:
        # ヒートマップ総和が0の場合は「重要領域なし」として信頼度=1.0で返す例
        return 1.0

    # 雨滴に隠されている"重要度"の総和
    overlap_val = np.sum(heatmap * raindrop_mask)
    coverage = overlap_val / heatmap_sum
    reliability = 1.0 - coverage
    return float(reliability)


def get_activation_hook(name):
    def hook(module, input, output):
        print(f"[HOOK CALLED] {name}")
        if name not in activations:
            activations[name] = []
        if len(activations[name]) >= 1:
            activations[name].clear()
        activations[name].append(output.detach().cpu())
    return hook
        
def tensor_to_heatmap(feature_map: torch.Tensor) -> np.ndarray:
    """
    feature_map: 形状が (C, H, W) のTensor (1枚のカメラ分)を想定。
    戻り値: ヒートマップ（カラー）のBGR画像 (H, W, 3)
    """
    # ---- 1. チャネル方向を平均して1枚にする (H, W)
    # 必要に応じて mean/max/sum などを切り替える
    heatmap = feature_map.mean(dim=0)  # -> shape: (H, W)

    # ---- 2. テンソル → numpy & 正規化
    heatmap = heatmap.detach().cpu().numpy()
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)

    # ---- 3. カラーマップを適用 (OpenCVはBGR)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def overlay_heatmap_on_image(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    image_bgr: 元画像 (H, W, 3)
    heatmap_bgr: ヒートマップ (H, W, 3)
    alpha: ヒートマップ重ねる割合
    戻り値: ヒートマップを重ね合わせたBGR画像
    """
    # サイズが違う場合は合わせる
    if (image_bgr.shape[0] != heatmap_bgr.shape[0]) or (image_bgr.shape[1] != heatmap_bgr.shape[1]):
        heatmap_bgr = cv2.resize(heatmap_bgr, (image_bgr.shape[1], image_bgr.shape[0]))

    # addWeighted でオーバーレイ
    overlaid = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlaid

def get_entry_point():
    return 'UniadAgent'

def add_raindrops_to_frame(frame, step=0, cam="CAM_FRONT"):
    
    pil_image = Image.fromarray(frame)

    # 動的に雨滴を生成、付与
    img_width, img_height = pil_image.size

    num_raindrops = 7  # 生成したい雨滴の数を指定

    moves_left  = [(-4, 0), (-4, +4), (0, +4)]   # 左側に沸いた雨滴 => 左か左下、あるいは真下へ
    moves_right = [(+4, 0), (+4, +4), (0, +4)]  # 右側に沸いた雨滴 => 右か右下、あるいは真下へ

    # ------------------------------------------------------------------------
    # (1) 初期スポーン: UniadAgent.RAINDROP_CACHEがNoneなら、雨滴をまとめて生成
    # ------------------------------------------------------------------------
    if UniadAgent.RAINDROP_CACHE is None:
        print("RAINDROP_CACHE is None -> spawning initial raindrops")
        UniadAgent.RAINDROP_CACHE = []
        
        # （a）まず1つは必ず中央にスポーン
        center_x = img_width // 2
        center_y = img_height // 2
        radius   = random.randint(120, 130)  # 適当
        # shape    = random.randint(0, 2)
        shape = 0
        side     = 'left' if center_x < (img_width // 2) else 'right'
        drop_center = Raindrop(
            key=1,
            centerxy=(center_x, center_y),
            radius=radius,
            shape=shape,
            side=side
        )
        UniadAgent.RAINDROP_CACHE.append(drop_center)

        # （b）残りはランダムにスポーン
        for i in range(num_raindrops - 1):
            key = i + 2
            # 例: 画像サイズに合わせて適当に乱数生成
            x = random.randint(0, img_width - 1)
            y = random.randint(0, img_height - 1)
            side = 'left' if x < (img_width // 2) else 'right'
            r   = random.randint(100, 110)
            shp = 0
            # shp = random.randint(0, 2)
            drop = Raindrop(key, (x, y), r, shape=shp, side=side)
            UniadAgent.RAINDROP_CACHE.append(drop)
    
    # ------------------------------------------------------------------------
    # (2) 既に雨滴リストがある場合 -> 移動＆画面外チェック
    # ------------------------------------------------------------------------
    else:
        print("RAINDROP_CACHE is not None -> update existing raindrops")
        
        updated_drops = []
        for drop in UniadAgent.RAINDROP_CACHE:
            side = drop.side
            if side == 'left':
                dx, dy = random.choice(moves_left)
            else:
                dx, dy = random.choice(moves_right)

            # 新しい中心へ
            new_center = (drop.center[0] + dx, drop.center[1] + dy)
            
            # 半径も少しランダム変動
            new_radius = drop.radius + random.randint(-1, 1)
            new_radius = max(10, min(new_radius, 200))  # 適当にクリップ

            # 画面内に残っているか
            if (0 <= new_center[0] < img_width) and (0 <= new_center[1] < img_height):
                # 画面内なら更新
                updated_drops.append(
                    Raindrop(drop.key, new_center, new_radius, shape=drop.shape, side=drop.side)
                )
            else:
                # 画像外に出たら消滅（appendしない）
                pass

        # 既存ドロップを更新
        UniadAgent.RAINDROP_CACHE = updated_drops

        # --------------------------------------------------------------------
        # (3) 雨滴が不足していたら追加スポーンして「num_raindrops個」になるように
        # --------------------------------------------------------------------
        current_count = len(UniadAgent.RAINDROP_CACHE)
        if current_count < num_raindrops:
            missing = num_raindrops - current_count
            print(f"{missing} raindrops are missing -> spawn new ones")

            # 不足分をまとめて生成（完全ランダム）
            # 必要に応じて「1つは中央」「1つは上部」など、再度ルールを設けてもOKです
            new_key_start = current_count + 1  # キーが衝突しないよう適当に決定
            for i in range(missing):
                key = new_key_start + i
                x = random.randint(0, img_width - 1)
                y = random.randint(0, img_height - 1)
                side = 'left' if x < (img_width // 2) else 'right'
                r   = random.randint(120, 130)
                # shp = random.randint(0, 2)
                shp = 0
                new_drop = Raindrop(key, (x, y), r, shape=shp, side=side)
                UniadAgent.RAINDROP_CACHE.append(new_drop)


    output_image, output_label, mask = generateDrops(pil_image, cfg, UniadAgent.RAINDROP_CACHE)


    from pathlib import Path
    
    base_save_dir = Path('/home/yoshi-22/Bench2Drive/output_cluster/output_images')
    base_save_dir.mkdir(parents=True, exist_ok=True)
    
    # カメラ名に対応するサブフォルダを定義
    camera_dirs = {
        "CAM_FRONT":       "front",
        "CAM_FRONT_LEFT":  "front_left",
        "CAM_FRONT_RIGHT": "front_right",
        "CAM_BACK":        "back",
        "CAM_BACK_LEFT":   "back_left",
        "CAM_BACK_RIGHT":  "back_right",
    }

    # 上記の定義にしたがってサブフォルダを選択 (cam が想定外なら "other" などにしてもよい)
    if cam in camera_dirs:
        folder_name = camera_dirs[cam]
    else:
        folder_name = "other"

    sub_dir = base_save_dir / folder_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名：例 「(RAINDROP_COUNT)_(frame).png」とする
    # UniadAgent は同ファイル内、もしくはインポート済み前提
    output_image_path = sub_dir / f"{UniadAgent.RAINDROP_COUNT}_{step}.png"

    # 保存
    output_image.save(output_image_path)
    # --------------------------------------------------------------------------------

    label_save_dir = "/home/yoshi-22/Bench2Drive/output_cluster/labels"
    os.makedirs(label_save_dir, exist_ok=True)
   
    if cam == "CAM_FRONT":
        output_label_path = os.path.join(label_save_dir, f"front_{UniadAgent.RAINDROP_COUNT}_{step}.png")
    elif cam == "CAM_FRONT_LEFT":
        output_label_path = os.path.join(label_save_dir, f"front_left_{UniadAgent.RAINDROP_COUNT}_{step}.png")
    elif cam == "CAM_FRONT_RIGHT":
        output_label_path = os.path.join(label_save_dir, f"front_right_{UniadAgent.RAINDROP_COUNT}_{step}.png")
    elif cam == "CAM_BACK":
        output_label_path = os.path.join(label_save_dir, f"back_{UniadAgent.RAINDROP_COUNT}_{step}.png")
    elif cam == "CAM_BACK_LEFT":
        output_label_path = os.path.join(label_save_dir, f"back_left_{UniadAgent.RAINDROP_COUNT}_{step}.png")
    elif cam == "CAM_BACK_RIGHT":
        output_label_path = os.path.join(label_save_dir, f"back_right_{UniadAgent.RAINDROP_COUNT}_{step}.png")
    else:
        output_label_path = os.path.join(label_save_dir, f"raindrop_label_{step}_{cam}.png")

    output_label_np = np.array(output_label)
    Image.fromarray((output_label_np * 255).astype(np.uint8)).save(output_label_path)

    # リサイズして出力
    resize_output_image = output_image.resize((1600, 900))
    output_array = np.array(resize_output_image)

    # 画像を黒
    output_array[:] = 0

    from pathlib import Path
    debug_dir = Path("/home/yoshi-22/Bench2Drive/output_cluster/debug_black")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_filename = debug_dir / f"black_debug_{cam}_{step:04d}.png"
    Image.fromarray(output_array).save(debug_filename)

    print(f"Saved black image: {debug_filename}")


    # if UniadAgent.RAINDROP_COUNT == 1:
    #   csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops1.csv"
    # elif UniadAgent.RAINDROP_COUNT == 2:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops2.csv"
    # elif UniadAgent.RAINDROP_COUNT == 3:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops3.csv"
    # elif UniadAgent.RAINDROP_COUNT == 4:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops4.csv"
    # elif UniadAgent.RAINDROP_COUNT == 5:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops5.csv"
    # elif UniadAgent.RAINDROP_COUNT == 6:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops6.csv"
    # elif UniadAgent.RAINDROP_COUNT == 7:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops7.csv"
    # elif UniadAgent.RAINDROP_COUNT == 8:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops8.csv"
    # elif UniadAgent.RAINDROP_COUNT == 9:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops9.csv"
    # else:
    #     csv_file_path = "/home/yoshi-22/RaindropsOnWindshield/raindrops_generator/csv_data/drops.csv"
    # raindrops = []

    # with open(csv_file_path, "r") as csv_file:
    #     reader = csv.DictReader(csv_file)
    #     for row in reader:
    #         key = int(row["Key"])
    #         center_x = int(row["CenterX"])
    #         center_y = int(row["CenterY"])
    #         radius = int(row["Radius"])
    #         shape = int(row["Shape"])
    #         raindrop = Raindrop(key, (center_x, center_y), radius, shape=shape)
    #         raindrops.append(raindrop)


    # output_image, _, mask = generateDrops(pil_image, cfg, raindrops)

    # # # 雨滴を画像に追加
    # # file_name = f"raindrop_{cam}_{UniadAgent.RAINDROP_COUNT}_{step}.png"

    # # out_folder, mask_folder = CAM_PATHS[cam]
    # # os.makedirs(out_folder, exist_ok=True)
    # # os.makedirs(mask_folder, exist_ok=True)
    # # save_path = os.path.join(out_folder, file_name)
    # # mask_path = os.path.join(mask_folder, file_name)
    # # output_image.save(save_path)
    # # mask.save(mask_path)

    # # リサイズして出力
    # resize_output_image = output_image.resize((1600, 900))
    # output_array = np.array(resize_output_image)

    return output_array

class UniadAgent(autonomous_agent.AutonomousAgent):
    # 初期設定
    # !!!!!ここがraindrop count
    RAINDROP_COUNT = -1
    RAINDROP_CACHE = None
    def setup(self, path_to_conf_file):
        # track.sensorsを設定することで、carlaがセンサーデータが必要なモードと判断
        self.track = autonomous_agent.Track.SENSORS
        # ステアリングのステップを設定
        self.steer_step = 0
        # 車両が動いているかどうか
        self.last_moving_status = 0
        # 最後に動いたステップ
        self.last_moving_step = -1
        self.last_steers = deque()
        self.pidcontroller = PIDController()
        # モデルの設定、ckptのパス
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            # ベンチマークをを実行する場合、設定ファイルの最後の要素を保存名として使用
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            # それ以外の場合、現在の日時を保存名として使用
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        # 設定ファイルの読み込み
        cfg = Config.fromfile(self.config_path)
        cfg.model['motion_head']['anchor_info_path'] = os.path.join('Bench2DriveZoo',cfg.model['motion_head']['anchor_info_path'])
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  

        # モデル構築
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        # チェックポイントの読み込み
        checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
        # gpuが使用可能な場合、モデルをgpuに移動
        self.model.cuda()
        # モデルを評価モードに設定
        self.model.eval()

        for name, module in self.model.named_modules():
            if name in valid_names:
                module.register_forward_hook(get_activation_hook(name))
                print(f'Registering hook for {name}')
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)
        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        # 画像データの前処理
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.last_steers = deque()
        # 初期化
        self.lat_ref, self.lon_ref = 42.0, 2.0
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control

        self.loaded_model = tf.keras.models.load_model("/home/yoshi-22/Bench2Drive/reliability/model.h5", compile=False)
        self.selector = joblib.load("/home/yoshi-22/Bench2Drive/reliability/anova_selector.pkl")
        self.freq_filters = create_freq_gabor_filter_bank(roi_size=ROI_SIZE, radii=GABOR_R, angles=GABOR_THETA)
        self.candidate_points = generate_grid_points(IMAGE_WIDTH, IMAGE_HEIGHT, ROI_SIZE, X_SPACING, Y_SPACING)

        self.populations = {}
        self.carrying_caps = {}
        # !!!!
        for cam_name in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            self.populations[cam_name] = None
            self.carrying_caps[cam_name] = INITIAL_POPULATION_SIZE

        # dirの作成
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            # string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string = self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()

            UniadAgent.RAINDROP_COUNT += 1
            print(f"RAINDROP_COUNT: {UniadAgent.RAINDROP_COUNT}")
   
        # write extrinsics directly
        # LiDARから画像座標への変換行列
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                                  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                       [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
         'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        # LiDARからカメラ座標への変換行列
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  ,  0.  , -1.  , -0.24],
                                 [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                      [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
          'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown
        # 座標系の統一

    def _init(self):
        
        try:
            # 地球座標（緯度・経度）とシミュレーション座標（x, y）の対応関係を求め、基準値として保持
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0        
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                
            ]
        
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        # カメラ画像の取得
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        
        # !!!!!!!雨滴付着
        # if UniadAgent.RAINDROP_COUNT >= 0:
        #     for cam, img in imgs.items():
        #         # OpenCV画像をPIL画像に変換
        #         img_with_raindrop = add_raindrops_to_frame(img, self.step, cam)
        #         imgs[cam] = img_with_raindrop

        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
    
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        # センサデータを取得
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        # カメラ画像を前処理し、resultsに格納
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
  
        # 車両角をquatに変換
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
  
        # 車両の位置や加速度などをcam_busに格納
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        command = tick_data['command_near']
        # defalutはlanefollow
        if command < 0:
            command = 4
        # モデルに入力するためにindexを１つ減らす
        command -= 1
        results['command'] = command
  
        theta_to_lidar = raw_theta
        # 次に経路店-車両位置をして、相対座標を計算
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        # 回転行列
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        # グローバル座標系から車両座標系に変換
        local_command_xy = rotation_matrix @ command_near_xy
  
        # LiDARから世界座標系への変換行列を計算
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        # 推論用パイプライン
        results = self.inference_only_pipeline(results)
        self.device="cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)
        # !!!!!!!推論結果を格納
        output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)

        for idx, cam in enumerate(['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']):
            img_bgr = tick_data['imgs'][cam]
            gray_resized = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (IMAGE_WIDTH, IMAGE_HEIGHT))
            # if self.populations[cam] is None:
            #     self.populations[cam] = initialize_population(INITIAL_POPULATION_SIZE)

            # updated_pop, updated_cup, labels = ea_process_single_frame(
            #     self.populations[cam],
            #     self.carrying_caps[cam],
            #     gray_resized,
            #     self.loaded_model,
            #     self.selector,
            #     self.freq_filters,
            #     self.candidate_points,
            #     self.step,
            #     out_dir = f"/home/yoshi-22/Bench2Drive/output_cluster/{cam}",
            # )


            # raindrop.cacheに基づいて雨滴マスクを作成
            if UniadAgent.RAINDROP_CACHE and len(UniadAgent.RAINDROP_CACHE) > 0:
                drop_labels = cluster_raindrops_dbscan(UniadAgent.RAINDROP_CACHE, eps=30.0, min_samples=1)

                raindrop_mask = build_raindrop_mask_from_cache(
                    raindrops = UniadAgent.RAINDROP_CACHE,
                    labels = drop_labels,
                    out_h = 512,
                    out_w = 900,
                    orig_w = 1600,
                    orig_h = 900
                )
                save_mask = (raindrop_mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(save_mask)
                mask_img.save(f"/home/yoshi-22/Bench2Drive/output_cluster/aaa/raindrop_mask_{self.step}.png")
                print(f"raindrop_mask.shape: {raindrop_mask.shape}")
            else:
                raindrop_mask = np.zeros((512,900), dtype=np.uint8)

            # if updated_pop:
            #     # raindrop mask作成
            #     # (ここでは 900×512 で扱うと仮定。カメラ画像が 900×1600 の場合は適宜変更)
            #     raindrop_mask = build_raindrop_mask(
            #         self.populations[cam], labels,
            #         image_h=gray_resized.shape[0],  # 512
            #         image_w=gray_resized.shape[1],  # 900
            #         roi_size=49
            #     )
            # else:
            #     raindrop_mask = np.zeros((512,900), dtype=np.uint8)

            min_avg_threshold = 0.4     # 過去平均が極端に小さい画素は「そもそも注目領域じゃない」
            ratio_threshold = 0.8

            single_feat_map = activations[valid_names[0]][0][idx]
            heatmap_2d = tensor_to_grayscale_map(single_feat_map, out_h = 512, out_w = 900)

            heatmap_hist[cam].append(heatmap_2d)

            if len(heatmap_hist[cam]) == HEATMAP_MEMORY_FRAMES:
                # # --- 過去平均ヒートマップ = M^{t-1} の近似
                # avg_heatmap = np.mean(heatmap_hist[cam], axis=0)
                # 5フレーム分の過去ヒートマップの中で最大値を代入
                avg_heatmap = np.max(heatmap_hist[cam], axis=0)

                print(f'aaaaaaaaaamask_avg_heatmap_max: {np.max(avg_heatmap)}')

                # (A) 雨滴がある & 過去平均がある程度高い画素
                mask_past_in_rain = (avg_heatmap >= min_avg_threshold) & (raindrop_mask == 1)
                num_pir = np.count_nonzero(mask_past_in_rain)
                print(f"mask_past_in_rain True count: {num_pir}")

                # (B) 「現在のヒートマップ M^t が 過去平均 M^{t-1} の ratio_threshold 倍以下」に落ち込んだ画素
                mask_diff_down = (heatmap_2d <= avg_heatmap * 0.8) & (raindrop_mask == 1)
                num_diff = np.count_nonzero(mask_diff_down)
                print(f"mask_diff_down True count: {num_diff}")

                # 上記(A) と (B) を満たす画素が「失われた領域」R^t_{Lost}
                R_t_lost = mask_past_in_rain & mask_diff_down
                num_lost = np.count_nonzero(R_t_lost)
                print(f"R_t_lost True count: {num_lost}")
                # -----------------------------
                # ★ (1) データ欠損度 D_lost^t の計算
                #     D_lost^t = Σ( M^{t-1}(x,y) - M^t(x,y) )  (x,y in R^t_lost)
                # -----------------------------
                # (avg_heatmap - heatmap_2d) を上記 R_t_lost に対して合計
                D_lost_value = np.sum((avg_heatmap - heatmap_2d)[R_t_lost])
                print(f"D_lost_value: {D_lost_value}")

                # (過去フレーム全体の注目度) = Σ M^{t-1}(x,y)
                # 必要に応じて，"avg_heatmap" を全画素で合計，あるいは特定の領域だけ合計など
                region_mask = (avg_heatmap >= min_avg_threshold) & (raindrop_mask == 1)
                sum_past = np.sum(avg_heatmap[R_t_lost])
                print(f"sum_past: {sum_past}")
                print(f"region_mask {D_lost_value / sum_past}")

                if sum_past > 1e-9:
                    # ★ (2) 信頼度
                    reliability_val = 1.0 - 0.838 * (D_lost_value / sum_past)
                else:
                    reliability_val = 1.0

            else:
                # まだ十分フレームがたまっていないとき -> 計算できないので信頼度=1.0 とする
                reliability_val = 1.0

            # ログ出力
            log_file_path = CAM_LOG_PATHS[cam]
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, "a") as f:
                f.write(
                    f"[{cam} step={self.step:04d}] reliability={reliability_val:.4f}\n"
                )

            # self.populations[cam] = updated_pop
            # self.carrying_caps[cam] = updated_cup
        
        print(f'aaaaaaaaalength of my_activations cus: {len(activations[valid_names[0]])}')

        out_truck =  output_data_batch[0]['planning']['result_planning']['sdc_traj'][0].cpu().numpy()
        # 経路結果を用いてpid制御
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        # 計算されたブレーキ量が0.05未満の場合、ブレーキをゼロに設定
        if brake_traj < 0.05: brake_traj = 0.0
        # スロットル（アクセル）がブレーキよりも大きい場合、ブレーキをゼロに設定
        if throttle_traj > brake_traj: brake_traj = 0.0
        # 現在の車速が5km/hを超える場合、スロットルをゼロに設定
        if tick_data['speed']>5:
            throttle_traj = 0
        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        print(f"now step", self.step)
        if SAVE_PATH is not None and self.step % 1 == 0:
            self.save(tick_data)
        self.prev_control = control
        return control

    def save(self, tick_data):
        #!!!!!!保存頻度
        # frame = self.step // 5
        # Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))

        # frame = self.step // 5
        frame = self.step

        base_save_path = pathlib.Path('/home/yoshi-22/Bench2Drive/output_cluster/output_heatmap')
        # 保存先フォルダ(例: rgb_front など)をあらかじめ作成しておく
        (base_save_path / 'rgb_front').mkdir(parents=True, exist_ok=True)
        (base_save_path / 'rgb_front_left').mkdir(parents=True, exist_ok=True)
        (base_save_path / 'rgb_front_right').mkdir(parents=True, exist_ok=True)
        (base_save_path / 'rgb_back').mkdir(parents=True, exist_ok=True)
        (base_save_path / 'rgb_back_left').mkdir(parents=True, exist_ok=True)
        (base_save_path / 'rgb_back_right').mkdir(parents=True, exist_ok=True)

        single_feat_map0 = activations[valid_names[0]][0][0]
        single_feat_map1 = activations[valid_names[0]][0][1]
        single_feat_map2 = activations[valid_names[0]][0][2]
        single_feat_map3 = activations[valid_names[0]][0][3]
        single_feat_map4 = activations[valid_names[0]][0][4]
        single_feat_map5 = activations[valid_names[0]][0][5]

        heatmap_bgr0 = tensor_to_heatmap(single_feat_map0)
        heatmap_bgr1 = tensor_to_heatmap(single_feat_map1)
        heatmap_bgr2 = tensor_to_heatmap(single_feat_map2)
        heatmap_bgr3 = tensor_to_heatmap(single_feat_map3)
        heatmap_bgr4 = tensor_to_heatmap(single_feat_map4)
        heatmap_bgr5 = tensor_to_heatmap(single_feat_map5)
        
        overlaid0 = overlay_heatmap_on_image(tick_data['imgs']['CAM_FRONT'], heatmap_bgr0, alpha=0.5)
        overlaid1 = overlay_heatmap_on_image(tick_data['imgs']['CAM_FRONT_RIGHT'], heatmap_bgr1, alpha=0.5)
        overlaid2 = overlay_heatmap_on_image(tick_data['imgs']['CAM_FRONT_LEFT'], heatmap_bgr2, alpha=0.5)
        overlaid3 = overlay_heatmap_on_image(tick_data['imgs']['CAM_BACK'], heatmap_bgr3, alpha=0.5)
        overlaid4 = overlay_heatmap_on_image(tick_data['imgs']['CAM_BACK_LEFT'], heatmap_bgr4, alpha=0.5)
        overlaid5 = overlay_heatmap_on_image(tick_data['imgs']['CAM_BACK_RIGHT'], heatmap_bgr5, alpha=0.5)

        overlaid0_rgb = cv2.cvtColor(overlaid0, cv2.COLOR_BGR2RGB)
        overlaid1_rgb = cv2.cvtColor(overlaid1, cv2.COLOR_BGR2RGB)
        overlaid2_rgb = cv2.cvtColor(overlaid2, cv2.COLOR_BGR2RGB)
        overlaid3_rgb = cv2.cvtColor(overlaid3, cv2.COLOR_BGR2RGB)
        overlaid4_rgb = cv2.cvtColor(overlaid4, cv2.COLOR_BGR2RGB)
        overlaid5_rgb = cv2.cvtColor(overlaid5, cv2.COLOR_BGR2RGB)

        overlaid0_pil = Image.fromarray(overlaid0_rgb)
        overlaid1_pil = Image.fromarray(overlaid1_rgb)
        overlaid2_pil = Image.fromarray(overlaid2_rgb)
        overlaid3_pil = Image.fromarray(overlaid3_rgb)
        overlaid4_pil = Image.fromarray(overlaid4_rgb)
        overlaid5_pil = Image.fromarray(overlaid5_rgb)

        overlaid0_pil.save(base_save_path / 'rgb_front' / ('front_%01d.png' % frame))
        overlaid1_pil.save(base_save_path / 'rgb_front_right' / ('front_right_%01d.png' % frame))
        overlaid2_pil.save(base_save_path / 'rgb_front_left' / ('front_left_%01d.png' % frame))
        overlaid3_pil.save(base_save_path / 'rgb_back' / ('back_%01d.png' % frame))
        overlaid4_pil.save(base_save_path / 'rgb_back_right' / ('back_right_%01d.png' % frame))
        overlaid5_pil.save(base_save_path / 'rgb_back_left' / ('back_left_%01d.png' % frame))

        base_save_dir = pathlib.Path('/home/yoshi-22/Bench2Drive/output_cluster/output_images')
        base_save_dir.mkdir(parents=True, exist_ok=True)
        # ディレクトリが存在しない場合は作成
        #　カメラ名に対応するフォルダ名をまとめて管理
        camera_dirs = {
            "CAM_FRONT": "front",
            "CAM_FRONT_LEFT": "front_left",
            "CAM_FRONT_RIGHT": "front_right",
            "CAM_BACK": "back",
            "CAM_BACK_LEFT": "back_left",
            "CAM_BACK_RIGHT": "back_right",
        }

        # # カメラ画像をそれぞれフォルダに分けて保存
        # for cam_key, folder_name in camera_dirs.items():
        #     sub_dir = base_save_dir / folder_name
        #     # ディレクトリが存在しない場合は作成
        #     sub_dir.mkdir(parents=True, exist_ok=True)

        #     # 保存ファイル名を作成 (カメラ別フォルダに配置)
        #     # 例: front/3_0012.png
        #     output_image_path = sub_dir / f"{UniadAgent.RAINDROP_COUNT}_{frame}.png"
            
        #     # tick_data['imgs'][cam_key] の画像を保存
        #     Image.fromarray(tick_data['imgs'][cam_key]).save(output_image_path)

        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    # gpsの座標をx,yに変換
    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
