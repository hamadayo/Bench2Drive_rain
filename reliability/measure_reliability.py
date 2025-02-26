import numpy as np
from scipy.spatial.distance import cosine
import cv2

# エッジヒストグラムの類似性を計算する関数
def calculate_edge_similarity(edge_histogram_prev, edge_histogram_current):
    """
    edge_histogram_prev: ndarray, 前フレームのエッジヒストグラム
    edge_histogram_current: ndarray, 現フレームのエッジヒストグラム
    return: float, コサイン類似度 (0から1)
    """
    similarity = 1 - cosine(edge_histogram_prev, edge_histogram_current)
    return max(0, similarity)  # コサイン類似度が負の場合を防ぐ

# 車両底部の視野角変化量を計算する関数
def calculate_view_angle_change(vehicle_bottom_angle_prev, vehicle_bottom_angle_current):
    """
    vehicle_bottom_angle_prev: float, 前フレームの車両底部の視野角
    vehicle_bottom_angle_current: float, 現フレームの車両底部の視野角
    return: float, 視野角の変化量
    """
    angle_change = abs(vehicle_bottom_angle_current - vehicle_bottom_angle_prev)
    return angle_change

# 信頼度を計算する関数
def calculate_monocular_reliability(edge_histogram_prev, edge_histogram_current,
                                    vehicle_bottom_angle_prev, vehicle_bottom_angle_current):
    """
    edge_histogram_prev: ndarray, 前フレームのエッジヒストグラム
    edge_histogram_current: ndarray, 現フレームのエッジヒストグラム
    vehicle_bottom_angle_prev: float, 前フレームの車両底部の視野角
    vehicle_bottom_angle_current: float, 現フレームの車両底部の視野角
    return: float, 信頼度 (0から1)
    """
    # エッジヒストグラムの類似性
    edge_similarity = calculate_edge_similarity(edge_histogram_prev, edge_histogram_current)

    # 車両底部の視野角変化量（正規化）
    max_angle_change = 10  # 視野角変化量の最大値（例）
    angle_change = calculate_view_angle_change(vehicle_bottom_angle_prev, vehicle_bottom_angle_current)
    normalized_angle_change = max(0, min(1, 1 - angle_change / max_angle_change))

    # 信頼度の統合
    reliability = (edge_similarity + normalized_angle_change) / 2
    return reliability

def calculate_edge_histogram(image, bins=8):
    """
    エッジヒストグラムを計算する関数
    image: 入力画像 (グレースケール)
    bins: ヒストグラムのビン数
    return: 正規化されたエッジヒストグラム
    ヒストぐらむのビン数は0~360度を8分割したもの.各区間のエッジの数をカウント
    """
    # エッジ検出 (Cannyを使用)
    edges = cv2.Canny(image, 100, 200)
    
    # 勾配方向を計算
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x方向の勾配
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y方向の勾配
    gradient_angle = np.arctan2(sobely, sobelx) * (180 / np.pi)  # 勾配方向を角度に変換
    gradient_angle[gradient_angle < 0] += 360  # 負の角度を正の範囲に変換
    
    # エッジの方向ヒストグラムを計算
    histogram, _ = np.histogram(gradient_angle[edges > 0], bins=bins, range=(0, 360))
    
    # ヒストグラムを正規化
    histogram = histogram / np.sum(histogram)
    return histogram

# 使用例
if __name__ == "__main__":
    # エッジヒストグラム（例）
    edge_histogram_prev = np.array([0.1, 0.2, 0.3, 0.4])
    edge_histogram_current = np.array([0.15, 0.25, 0.35, 0.45])

    # 車両底部の視野角（例）
    vehicle_bottom_angle_prev = 2.5  # 前フレームの視野角
    vehicle_bottom_angle_current = 2.7  # 現フレームの視野角

    # 信頼度を計算
    reliability = calculate_monocular_reliability(edge_histogram_prev, edge_histogram_current,
                                                  vehicle_bottom_angle_prev, vehicle_bottom_angle_current)

    print(f"Monocular reliability: {reliability:.2f}")
