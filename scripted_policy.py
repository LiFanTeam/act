import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


class StackCubePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_phase = 'approach_red'
        self.right_phase = 'wait'
        self.left_phase_steps = 0
        self.right_phase_steps = 0
        self.stack_center = np.array([0.0, 0.5, 0.05])
        self.left_init_pose = None
        self.right_init_pose = None
        self.left_pick_quat = None
        self.right_pick_quat = None
        self.left_support_quat = None
        self.current_left_gripper = None
        self.current_right_gripper = None
        self.gripper_delta = 0.08

    @staticmethod
    def _step_towards(curr_xyz, target_xyz, max_delta):
        delta = target_xyz - curr_xyz
        distance = np.linalg.norm(delta)
        if distance <= max_delta or distance == 0:
            return target_xyz.copy()
        return curr_xyz + delta / distance * max_delta

    @staticmethod
    def _step_gripper(curr, target, max_delta):
        delta = np.clip(target - curr, -max_delta, max_delta)
        return float(np.clip(curr + delta, 0.0, 1.0))

    @staticmethod
    def _is_close(curr_xyz, target_xyz, threshold):
        return np.linalg.norm(curr_xyz - target_xyz) < threshold

    def _transition_left(self, new_phase):
        self.left_phase = new_phase
        self.left_phase_steps = 0

    def _transition_right(self, new_phase):
        self.right_phase = new_phase
        self.right_phase_steps = 0

    def _compute_left_target(self, obs, red_pose):
        curr_pose = obs['mocap_pose_left']
        curr_xyz = curr_pose[:3]
        red_xyz = red_pose[:3]
        normal_step = 0.018
        slow_step = 0.006

        if self.left_phase == 'approach_red':
            target_xyz = red_xyz + np.array([0.0, 0.0, 0.08])
            target_quat = self.left_pick_quat
            gripper = 1.0
            step_size = normal_step
            if self._is_close(curr_xyz, target_xyz, 0.02):
                self._transition_left('grasp_red')
        elif self.left_phase == 'grasp_red':
            target_xyz = red_xyz + np.array([0.0, 0.0, -0.015])
            target_quat = self.left_pick_quat
            gripper = 1.0
            step_size = slow_step
            if self._is_close(curr_xyz, target_xyz, 0.01):
                self._transition_left('close_red')
        elif self.left_phase == 'close_red':
            target_xyz = red_xyz + np.array([0.0, 0.0, -0.015])
            target_quat = self.left_pick_quat
            gripper = 0.0
            step_size = slow_step
            if self.left_phase_steps >= 15:
                self._transition_left('lift_red')
        elif self.left_phase == 'lift_red':
            target_xyz = red_xyz + np.array([0.0, 0.0, 0.10])
            target_quat = self.left_pick_quat
            gripper = 0.0
            step_size = normal_step
            if red_xyz[2] > 0.09:
                self._transition_left('move_red_center')
            elif self.left_phase_steps >= 40 and red_xyz[2] < 0.045:
                self._transition_left('approach_red')
        elif self.left_phase == 'move_red_center':
            target_xyz = np.array([-0.02, self.stack_center[1], 0.12])
            target_quat = self.left_pick_quat
            gripper = 0.0
            step_size = normal_step
            if self._is_close(curr_xyz, target_xyz, 0.025):
                self._transition_left('place_red')
        elif self.left_phase == 'place_red':
            target_xyz = np.array([-0.01, self.stack_center[1], 0.04])
            target_quat = self.left_pick_quat
            gripper = 0.0
            step_size = slow_step
            centered = np.linalg.norm(red_xyz[:2] - self.stack_center[:2]) < 0.035
            if centered and red_xyz[2] < 0.065 and self.left_phase_steps >= 20:
                self._transition_left('release_red')
        elif self.left_phase == 'release_red':
            target_xyz = np.array([-0.02, self.stack_center[1], 0.14])
            target_quat = self.left_pick_quat
            gripper = 1.0
            step_size = slow_step
            if self.left_phase_steps >= 15:
                self._transition_left('support_red')
        else:
            target_xyz = np.array([-0.20, self.stack_center[1], 0.22])
            target_quat = self.left_pick_quat
            gripper = 1.0
            step_size = normal_step

        next_xyz = self._step_towards(curr_xyz, target_xyz, step_size)
        self.left_phase_steps += 1
        return next_xyz, target_quat, gripper

    def _compute_right_target(self, obs, red_pose, blue_pose):
        curr_pose = obs['mocap_pose_right']
        curr_xyz = curr_pose[:3]
        red_xyz = red_pose[:3]
        blue_xyz = blue_pose[:3]
        normal_step = 0.018
        slow_step = 0.004

        if self.right_phase == 'wait':
            target_xyz = self.right_init_pose[:3]
            target_quat = self.right_pick_quat
            gripper = 1.0
            step_size = normal_step
            if self.left_phase == 'support_red' and self.left_phase_steps >= 10:
                self._transition_right('approach_blue')
        elif self.right_phase == 'approach_blue':
            target_xyz = blue_xyz + np.array([0.0, 0.0, 0.08])
            target_quat = self.right_pick_quat
            gripper = 1.0
            step_size = normal_step
            if self._is_close(curr_xyz, target_xyz, 0.02):
                self._transition_right('grasp_blue')
        elif self.right_phase == 'grasp_blue':
            target_xyz = blue_xyz + np.array([0.0, 0.0, -0.015])
            target_quat = self.right_pick_quat
            gripper = 1.0
            step_size = slow_step
            if self._is_close(curr_xyz, target_xyz, 0.01):
                self._transition_right('close_blue')
        elif self.right_phase == 'close_blue':
            target_xyz = blue_xyz + np.array([0.0, 0.0, -0.015])
            target_quat = self.right_pick_quat
            gripper = 0.0
            step_size = slow_step
            if self.right_phase_steps >= 15:
                self._transition_right('lift_blue')
        elif self.right_phase == 'lift_blue':
            target_xyz = blue_xyz + np.array([0.0, 0.0, 0.12])
            target_quat = self.right_pick_quat
            gripper = 0.0
            step_size = normal_step
            if blue_xyz[2] > 0.09:
                self._transition_right('align_stack')
            elif self.right_phase_steps >= 40 and blue_xyz[2] < 0.045:
                self._transition_right('approach_blue')
        elif self.right_phase == 'align_stack':
            target_xyz = np.array([red_xyz[0], red_xyz[1], red_xyz[2] + 0.12])
            target_quat = self.right_pick_quat
            gripper = 0.0
            step_size = normal_step
            xy_close = np.linalg.norm(curr_xyz[:2] - target_xyz[:2]) < 0.015
            z_close = abs(curr_xyz[2] - target_xyz[2]) < 0.02
            if xy_close and z_close:
                self._transition_right('descend_stack')
        elif self.right_phase == 'descend_stack':
            target_xyz = np.array([red_xyz[0], red_xyz[1], red_xyz[2] + 0.075])
            target_quat = self.right_pick_quat
            gripper = 0.0
            step_size = slow_step
            if self._is_close(curr_xyz, target_xyz, 0.008):
                self._transition_right('release_blue')
        elif self.right_phase == 'release_blue':
            target_xyz = np.array([red_xyz[0], red_xyz[1], red_xyz[2] + 0.078])
            target_quat = self.right_pick_quat
            gripper = 1.0
            step_size = slow_step
            if self.right_phase_steps >= 25:
                self._transition_right('retreat')
        else:
            target_xyz = np.array([0.18, 0.46, 0.20])
            target_quat = self.right_pick_quat
            gripper = 1.0
            step_size = normal_step

        next_xyz = self._step_towards(curr_xyz, target_xyz, step_size)
        self.right_phase_steps += 1
        return next_xyz, target_quat, gripper

    def __call__(self, ts):
        obs = ts.observation
        red_pose = np.array(obs['env_state'][:7])
        blue_pose = np.array(obs['env_state'][7:])

        if self.step_count == 0:
            self.left_init_pose = obs['mocap_pose_left'].copy()
            self.right_init_pose = obs['mocap_pose_right'].copy()
            self.left_pick_quat = (
                Quaternion(self.left_init_pose[3:])
                * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)
            ).elements
            self.right_pick_quat = (
                Quaternion(self.right_init_pose[3:])
                * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
            ).elements
            self.left_support_quat = Quaternion(
                axis=[1.0, 0.0, 0.0],
                degrees=90,
            ).elements
            self.current_left_gripper = float(obs['qpos'][6])
            self.current_right_gripper = float(obs['qpos'][13])

        left_xyz, left_quat, left_gripper = self._compute_left_target(obs, red_pose)
        right_xyz, right_quat, right_gripper = self._compute_right_target(
            obs,
            red_pose,
            blue_pose,
        )

        self.current_left_gripper = self._step_gripper(
            self.current_left_gripper,
            left_gripper,
            self.gripper_delta,
        )
        self.current_right_gripper = self._step_gripper(
            self.current_right_gripper,
            right_gripper,
            self.gripper_delta,
        )

        if self.inject_noise:
            scale = 0.002
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        self.step_count += 1
        return np.concatenate(
            [
                np.concatenate([left_xyz, left_quat, [self.current_left_gripper]]),
                np.concatenate([right_xyz, right_quat, [self.current_right_gripper]]),
            ]
        )


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
        policy_cls = PickAndTransferPolicy
    elif 'sim_stack_cube' in task_name:
        env = make_ee_sim_env('sim_stack_cube_scripted')
        policy_cls = StackCubePolicy
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
        policy_cls = InsertionPolicy
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = policy_cls(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)
