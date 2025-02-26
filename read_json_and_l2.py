import json
import math

# # --- JSONファイルを読み込む ---
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_0_rep0_Town12_ParkingCutIn_1_15_02_02_05_25_32/metric_info.json', 'r', encoding='utf-8') as f1:
#     data1 = json.load(f1)

# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_1_rep0_Town12_ParkingCutIn_1_15_02_02_06_26_45/metric_info.json', 'r', encoding='utf-8') as f2:
#     data2 = json.load(f2)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_2_rep0_Town12_ParkingCutIn_1_15_02_02_07_40_08/metric_info.json', 'r', encoding='utf-8') as f3:
#     data3 = json.load(f3)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_3_rep0_Town12_ParkingCutIn_1_15_02_02_08_58_54/metric_info.json', 'r', encoding='utf-8') as f4:
#     data4 = json.load(f4)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_4_rep0_Town12_ParkingCutIn_1_15_02_02_10_18_43/metric_info.json', 'r', encoding='utf-8') as f5:
#     data5 = json.load(f5)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_5_rep0_Town12_ParkingCutIn_1_15_02_02_11_30_47/metric_info.json', 'r', encoding='utf-8') as f6:
#     data6 = json.load(f6)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_6_rep0_Town12_ParkingCutIn_1_15_02_02_12_44_19/metric_info.json', 'r', encoding='utf-8') as f7:
#     data7 = json.load(f7)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_7_rep0_Town12_ParkingCutIn_1_15_02_02_13_55_56/metric_info.json', 'r', encoding='utf-8') as f8:
#     data8 = json.load(f8)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_8_rep0_Town12_ParkingCutIn_1_15_02_02_15_07_28/metric_info.json', 'r', encoding='utf-8') as f9:
#     data9 = json.load(f9)
# with open('/media/yoshi-22/T7/closed_loop/straight/output_cluster/RouteScenario_9_rep0_Town12_ParkingCutIn_1_15_02_02_16_17_59/metric_info.json', 'r', encoding='utf-8') as f10:
#     data10 = json.load(f10)


# --- JSONファイルを読み込む ---
# with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_0_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_01_20_36_04/metric_info.json', 'r', encoding='utf-8') as f1:
#     data1 = json.load(f1)

# with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_1_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_01_20_58_47/metric_info.json', 'r', encoding='utf-8') as f2:
#     data2 = json.load(f2)
# with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_2_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_01_21_28_32/metric_info.json', 'r', encoding='utf-8') as f3:
#     data3 = json.load(f3)
# with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_3_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_01_22_01_54/metric_info.json', 'r', encoding='utf-8') as f4:
#     data4 = json.load(f4)
# with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_4_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_01_22_36_15/metric_info.json', 'r', encoding='utf-8') as f5:
#     data5 = json.load(f5)
# with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_5_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_01_23_06_12/metric_info.json', 'r', encoding='utf-8') as f6:
#     data6 = json.load(f6)
# with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_6_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_01_23_36_38/metric_info.json', 'r', encoding='utf-8') as f7:
#     data7 = json.load(f7)
with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_7_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_02_00_09_29/metric_info.json', 'r', encoding='utf-8') as f8:
    data8 = json.load(f8)
with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_8_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_02_00_37_08/metric_info.json', 'r', encoding='utf-8') as f9:
    data9 = json.load(f9)
with open('/media/yoshi-22/T7/closed_loop/turn_left/output_cluster/RouteScenario_9_rep0_Town13_VanillaNonSignalizedTurnEncounterStopsign_1_0_02_02_01_04_59/metric_info.json', 'r', encoding='utf-8') as f10:
    data10 = json.load(f10)


# --- JSONファイルを読み込む ---
with open('/media/yoshi-22/T7/closed_loop/turn_right/output_cluster0~6/RouteScenario_0_rep0_Town15_SignalizedJunctionRightTurn_1_18_01_31_23_32_22/metric_info.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open('/media/yoshi-22/T7/closed_loop/turn_right/output_cluster0~6/RouteScenario_1_rep0_Town15_SignalizedJunctionRightTurn_1_18_02_01_00_05_02/metric_info.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)
with open('/media/yoshi-22/T7/closed_loop/turn_right/output_cluster0~6/RouteScenario_2_rep0_Town15_SignalizedJunctionRightTurn_1_18_02_01_00_46_05/metric_info.json', 'r', encoding='utf-8') as f3:
    data3 = json.load(f3)
with open('/media/yoshi-22/T7/closed_loop/turn_right/output_cluster0~6/RouteScenario_3_rep0_Town15_SignalizedJunctionRightTurn_1_18_02_01_01_31_52/metric_info.json', 'r', encoding='utf-8') as f4:
    data4 = json.load(f4)
with open('/media/yoshi-22/T7/closed_loop/turn_right/output_cluster0~6/RouteScenario_4_rep0_Town15_SignalizedJunctionRightTurn_1_18_02_01_02_17_55/metric_info.json', 'r', encoding='utf-8') as f5:
    data5 = json.load(f5)
with open('/media/yoshi-22/T7/closed_loop/turn_right/output_cluster0~6/RouteScenario_5_rep0_Town15_SignalizedJunctionRightTurn_1_18_02_01_03_01_08/metric_info.json', 'r', encoding='utf-8') as f6:
    data6 = json.load(f6)
with open('/media/yoshi-22/T7/closed_loop/turn_right/output_cluster7~9/RouteScenario_0_rep0_Town15_SignalizedJunctionRightTurn_1_18_02_01_15_59_04/metric_info.json', 'r', encoding='utf-8') as f7:
    data7 = json.load(f7)
# with open('', 'r', encoding='utf-8') as f8:
#     data8 = json.load(f8)
# with open('', 'r', encoding='utf-8') as f9:
#     data9 = json.load(f9)
# with open('', 'r', encoding='utf-8') as f10:
#     data10 = json.load(f10)



# 比較したいファイルをリスト化（タプルの2要素目はわかりやすく番号を付けておく）
files_to_compare = [
    (data2, 2),
    (data3, 3),
    (data4, 4),
    (data5, 5),
    (data6, 6),
    (data7, 7),
    (data8, 8),
    (data9, 9),
    (data10, 10),
]

# f1と各ファイルを個別に比較して、合計距離・平均距離を出す
for dataX, idx in files_to_compare:
    # f1のキーと dataX のキーに共通するものだけを対象に
    common_keys = set(data1.keys()).intersection(set(dataX.keys()))
    
    distance_sum = 0.0
    count = 0
    
    for key in common_keys:
        loc1 = data1[key]['location']  # [x1, y1, z1]
        loc2 = dataX[key]['location']  # [x2, y2, z2]
        
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        dz = loc1[2] - loc2[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        distance_sum += dist
        count += 1
    
    # 結果の出力
    if count > 0:
        average_dist = distance_sum / count
        print(f"f1 と f{idx} の比較結果:")
        print(f"  共通フレーム数: {count}")
        print(f"  距離合計: {distance_sum:.4f}")
        print(f"  平均距離(L2ノルム): {average_dist:.4f}")
    else:
        print(f"f1 と f{idx} の共通キーがなく、距離を計算できませんでした。")
