import cv2
import numpy as np
import glob
import os

def gradient_based_raindrop_detection_bbox(
    frames,
    Tb=0.18,           # 閾値(論文内例)
    gauss_kernel=27,   # ガウシアンフィルタサイズ
    dilation_kernel=9, # 膨張演算のカーネルサイズ
    Td=0.1             # 雨滴が画像を覆う割合の閾値
):
    """
    勾配ベースの雨滴検出(過去Nフレーム平均)を行い、
    最終的にバイナリマスクと、それに対応するバウンディングボックスのリストを返す。

    Parameters
    ----------
    frames : list of np.ndarray
        入力フレーム(グレースケールまたはBGR)を N 枚まとめたもの。
        ※今回は常にN=10枚を想定(足りない場合は補完)。
    Tb : float
        勾配マップを反転二値化する際の閾値(0〜1換算)。
    gauss_kernel : int
        ガウシアンフィルタのカーネルサイズ(奇数)。
    dilation_kernel : int
        二値マスクに対する膨張演算のカーネルサイズ(奇数推奨)。
    Td : float
        「雨滴が存在する」とみなす最終判定のための白画素割合しきい値(0〜1)。

    Returns
    -------
    bin_mask : np.ndarray
        2値マスク画像(0 or 255) [H, W]。
    bbox_list : list of tuple
        マスク領域を取り囲む外接矩形(Bounding Box)のリスト。
        [(x, y, w, h), ...] の形。x,yは左上座標。w,hは幅・高さ。
    raindrop_ratio : float
        bin_mask内で白になっているピクセルの割合(0〜1)。
    has_raindrops : bool
        raindrop_ratio > Td であれば True、それ以外は False。
    """

    if not frames:
        # フレームが空リストなら処理不可なのでNone返す
        return None, [], 0.0, False

    # フレーム枚数
    N = len(frames)

    # 全てグレースケールに変換
    gray_frames = []
    for f in frames:
        if len(f.shape) == 3:
            gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        else:
            gray_frames.append(f)

    # Sobel勾配マップを各フレームで計算 → 平均
    gradient_maps = []
    for gf in gray_frames:
        Gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(Gx, Gy)
        gradient_maps.append(grad)

    # フレーム間で平均化
    stacked = np.stack(gradient_maps, axis=2)  # shape: (H, W, N)
    avg_gradient_map = np.mean(stacked, axis=2)  # shape: (H, W)

    # ガウシアンフィルタでノイズ除去
    if gauss_kernel > 1:
        avg_gradient_map = cv2.GaussianBlur(avg_gradient_map, (gauss_kernel, gauss_kernel), 0)

    # 0〜255にクリップ＆8bit変換
    norm_map = np.clip(avg_gradient_map, 0, 255).astype(np.uint8)

    # 逆二値化 (勾配が Tb より小さい部分を白=255)
    ret, bin_mask = cv2.threshold(
        norm_map,
        int(255 * Tb),
        255,
        cv2.THRESH_BINARY_INV
    )

    # 膨張 (dilate)
    if dilation_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
        bin_mask = cv2.dilate(bin_mask, kernel, iterations=1)

    # 白画素割合を計算
    H, W = bin_mask.shape
    white_pixels = np.count_nonzero(bin_mask == 255)
    raindrop_ratio = white_pixels / (H * W)

    # 判定
    has_raindrops = (raindrop_ratio > Td)

    # --- バウンディングボックスの取得 ---
    # 白領域(=雨滴領域)の輪郭を取得
    contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_list.append((x, y, w, h))

    return bin_mask, bbox_list, raindrop_ratio, has_raindrops


def replicate_last_frame_if_needed(frame_buffer, required_frames=10):
    """
    frame_bufferの長さがrequired_framesに満たない場合は、
    最後のフレームを複製して補う。十分あればそのまま返す。
    """
    frames_needed = required_frames - len(frame_buffer)
    if frames_needed > 0 and len(frame_buffer) > 0:
        last_frame = frame_buffer[-1]
        for _ in range(frames_needed):
            frame_buffer.append(last_frame.copy())
    return frame_buffer


def main(
    image_folder="images",   # 入力画像が連番で入っているフォルダ
    Tb=0.18,
    gauss_kernel=27,
    dilation_kernel=9,
    Td=0.1,
    required_frames=10
):
    """
    指定フォルダから画像を読み込み、常に「過去N枚(=required_frames)」を用いて
    雨滴検出(勾配ベース+バウンディングボックス出力)を行うサンプル。
    """

    # 1) 画像ファイル一覧をソートして取得 (jpg/pngなど)
    #    glob.globのパターンはフォルダ構成に合わせて調整してください
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not image_paths:
        print("No images found in folder:", image_folder)
        return

    # 過去フレームをためるためのバッファ
    frame_buffer = []

    # 読み込んだ画像1枚ごとに処理
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        # バッファに追加
        frame_buffer.append(img)

        # バッファが required_frames より多くなったら古いのを捨てる
        if len(frame_buffer) > required_frames:
            frame_buffer.pop(0)  # 先頭を削除

        # 現時点で何枚たまっているか
        current_count = len(frame_buffer)

        # 十分フレームが集まっていない場合も、最後のフレームを複製して補完する一例
        tmp_frames = replicate_last_frame_if_needed(frame_buffer.copy(), required_frames)
        # ↑ ここを「もし足りないならスキップ」したい場合はコメントアウトし、
        #    required_frames未満なら何もしない、といった条件にする。

        # 雨滴検出
        bin_mask, bbox_list, ratio, has_drop = gradient_based_raindrop_detection_bbox(
            tmp_frames,
            Tb=Tb,
            gauss_kernel=gauss_kernel,
            dilation_kernel=dilation_kernel,
            Td=Td
        )

        print(f"[{idx+1}/{len(image_paths)}] {os.path.basename(path)}")
        print(f"  -> raindrop_ratio = {ratio:.3f}, has_drop = {has_drop}")
        print(f"  -> bounding boxes = {bbox_list}")

        # ------------ 結果表示用 ------------
        # マスクをカラー化してバウンディングボックスを描画してみる
        mask_bgr = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)

        # bboxを描画(赤色)
        for (x, y, w, h) in bbox_list:
            cv2.rectangle(mask_bgr, (x, y), (x+w, y+h), (0,0,255), 2)

        # 表示用に並べる(元画像とマスク画像を横にconcat)
        display_img = np.hstack([img, mask_bgr])
        cv2.imshow("Raindrop Detection - Press ESC to exit", display_img)
        k = cv2.waitKey(500)  # 0.5秒表示
        if k == 27:  # ESCキー
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(
        image_folder="images",   # 画像ファイルが入っているパスを指定
        Tb=0.18,
        gauss_kernel=27,
        dilation_kernel=9,
        Td=0.1,
        required_frames=10
    )
