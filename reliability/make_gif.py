import os
from PIL import Image

def create_gif_from_folder(
    input_folder,
    output_gif_path,
    fps=10,
    valid_extensions=(".png", ".jpg", ".jpeg")
):
    """
    指定したフォルダ内の画像をファイル名でソートし、アニメーションGIFを作成する。
    
    Parameters:
    -----------
    input_folder : str
        画像が格納されているフォルダのパス
    output_gif_path : str
        出力するGIFファイルのパス (例: "/path/to/output.gif")
    fps : int, optional
        1秒あたり何枚の画像を表示するか (初期値=10)
    valid_extensions : tuple, optional
        対象とする画像拡張子のタプル (初期値=(".png", ".jpg", ".jpeg"))
    """

    # フォルダ内の画像ファイル一覧を取得し、ファイル名でソート
    file_list = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(valid_extensions)
    ]
    file_list.sort()

    if not file_list:
        print("指定フォルダに画像がありません。")
        return

    # 画像を読み込んでリスト化
    images = []
    for filename in file_list:
        path = os.path.join(input_folder, filename)
        img = Image.open(path)
        images.append(img)

    # fpsをもとに、1コマあたりのミリ秒(duration)を計算
    duration_ms = int(1000 / fps)

    # 1枚目の画像をベースにsaveし、残りをappend_imagesで追加
    # loop=0 は「無限ループ」を意味する
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"GIFを作成しました: {output_gif_path}")


if __name__ == "__main__":
    # 例: "sample_images" フォルダにある画像をGIFにまとめる
    input_folder_path = "/home/yoshi-22//UniAD/outputs_img_backbone.layer4.2/CAM_FRONT"
    output_gif_file = "/home/yoshi-22/Bench2Drive/reliability/output.gif"
    create_gif_from_folder(input_folder_path, output_gif_file, fps=5)
