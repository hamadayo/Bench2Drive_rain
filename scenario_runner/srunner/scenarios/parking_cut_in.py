#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Parking cut in scenario synchronizes a vehicle that is parked at a side lane
to cut in in front of the ego vehicle, forcing it to break
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      BasicAgentBehavior)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, ChangeRoadBehavior


class ParkingCutIn(BasicScenario):

    """
    Parking cut in scenario synchronizes a vehicle that is parked at a side lane
    to cut in in front of the ego vehicle, forcing it to break
    路肩に停車中の車両が自車の前に割り込むシナリオ。
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        print("======ParkingCutIn========")
        # マップデータ、トリガーポイントの開始地点、waypointの取得
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)

        # 車両の割り込む位置とブロックする車両の位置を設定
        self._cut_in_distance = 35
        self._blocker_distance = 28

        # 割り込む車両の速度、自社の反応するまでの時間、トリガー発生の最小距離、割り込みが終了する距離、割り込みに必要な追加スペースを設定
        self._adversary_speed = 13.0  # Speed of the adversary [m/s]
        self._reaction_time = 2.35  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 10.0  # Min distance to the collision location that triggers the adversary [m]
        self._end_distance = 600
        self._extra_space = 20

        # 車両の生成条件
        self._bp_attributes = {'base_type': 'car', 'generation': 2, 'special_type': ''}

        self.timeout = timeout

        # 駐車車両の位置を左右どちらか設定
        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = "right"

        super().__init__("ParkingCutIn",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Spawn the blocker vehicle
        # blockする車両の位置を取得
        blocker_wps = self._reference_waypoint.next(self._blocker_distance)
        if not blocker_wps:
            raise ValueError("Couldn't find a proper position for the cut in vehicle")
        self._blocker_wp = blocker_wps[0]

        if self._direction == 'left':
            parking_wp = self._blocker_wp.get_left_lane()
        else:
            parking_wp = self._blocker_wp.get_right_lane()

        self.parking_slots.append(parking_wp.transform.location)

        self._blocker_actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', parking_wp.transform, 'scenario no lights', attribute_filter={'base_type': 'car', 'generation': 2})
        if not self._blocker_actor:
            raise ValueError("Couldn't spawn the parked actor")
        self._blocker_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(self._blocker_actor)

        side_location = self._get_displaced_location(self._blocker_actor, parking_wp)
        self._blocker_actor.set_location(side_location)

        collision_wps = self._reference_waypoint.next(self._cut_in_distance)
        if not collision_wps:
            raise ValueError("Couldn't find a proper position for the cut in vehicle")
        self._collision_wp = collision_wps[0]

        # Get the parking direction
        if self._direction == 'left':
            parking_wp = self._collision_wp.get_left_lane()
        else:
            parking_wp = self._collision_wp.get_right_lane()

        self.parking_slots.append(parking_wp.transform.location)

        # parking_wp の位置に駐車車両を生成
        self._parked_actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', parking_wp.transform, 'scenario', attribute_filter=self._bp_attributes)
        if not self._parked_actor:
            raise ValueError("Couldn't spawn the parked actor")
        self.other_actors.append(self._parked_actor)

        # 道路の端に移動させるための新しい位置を計算し、駐車車両を移動
        side_location = self._get_displaced_location(self._parked_actor, parking_wp)
        self._parked_actor.set_location(side_location)

    def _get_displaced_location(self, actor, wp):
        """
        Calculates the location such that the actor is at the sidemost part of the lane
        """
        # Move the actor to the edge of the lane near the sidewalk
        displacement = (wp.lane_width - actor.bounding_box.extent.y) / 4
        displacement_vector = wp.transform.get_right_vector()
        if self._direction == 'left':
            displacement_vector *= -1

        new_location = wp.transform.location + carla.Location(x=displacement*displacement_vector.x,
                                                              y=displacement*displacement_vector.y,
                                                              z=displacement*displacement_vector.z)
        new_location.z += 0.05  # Just in case, avoid collisions with the ground
        return new_location

    def _create_behavior(self):
        """
        After invoking this scenario, a parked vehicle will wait for the ego to
        be close-by, merging into its lane, forcing it to break.
        """
        # self.other_actors[0]: ブロック車両 (self._blocker_actor)
        # self.other_actors[1]: 割り込み車両 (self._parked_actor
        # 動作ツリーの「シーケンスノード」を作成
        # 子ノードを順番に実行し、すべての子が成功すると親も成功となる
        sequence = py_trees.composites.Sequence(name="ParkingCutIn")
        if self.route_mode:
            # 自車と他車両の間に一定の距離（self._cut_in_distance + 10）を確保する動作を追加
            sequence.add_child(LeaveSpaceInFront(self._cut_in_distance + 10))

        # 割り込み地点のウェイポイント
        collision_location = self._collision_wp.transform.location

        # Wait until ego is close to the adversary
        # 以下の子ノードのうち1つでも成功したら、ノード全体が成功
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        # 時車が割り込み地点に到達するまでの時間がreaction_time以下なら成功
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        # 時車が割り込み地点の一定の距離以内に入ったら成功
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        # 割り込みを待つ動作をシーケンスに追加
        sequence.add_child(trigger_adversary)

        # 自車の進行を調整するため、道路の振る舞いを変更する動作を追加
        if self.route_mode:
            sequence.add_child(ChangeRoadBehavior(extra_space=self._extra_space))

        # Move the adversary
        cut_in = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="Cut in behavior")
        # 割り込み車両の動作を定義
        cut_in.add_child(BasicAgentBehavior(self.other_actors[1], opt_dict={'ignore_traffic_lights': True, 'target_speed': self._adversary_speed}))
        # もしくは一定距離進む
        cut_in.add_child(DriveDistance(self.other_actors[1], self._end_distance))
        sequence.add_child(cut_in)

        # 割り込み後の継続動作を追加
        continue_driving = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
            name="Continue Driving After Cut In"
        )
        continue_driving.add_child(DriveDistance(self.other_actors[1], 500))  # 50m進む
        sequence.add_child(continue_driving)

        # Remove everything
        # 割り込み後、アクターを削除
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyAdversary"))
        sequence.add_child(ActorDestroy(self.other_actors[1], name="DestroyBlocker"))

        # 車両間の追加スペースを削除する
        if self.route_mode:
            sequence.add_child(ChangeRoadBehavior(extra_space=0))

        return sequence

    # 衝突が発生したか判定
    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
