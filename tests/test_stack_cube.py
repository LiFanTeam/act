import unittest

import numpy as np


class StackCubeTaskTests(unittest.TestCase):
    def test_task_config_exists(self):
        from constants import SIM_TASK_CONFIGS

        self.assertIn("sim_stack_cube_scripted", SIM_TASK_CONFIGS)
        self.assertEqual(
            SIM_TASK_CONFIGS["sim_stack_cube_scripted"]["camera_names"],
            ["top", "angle"],
        )
        self.assertEqual(SIM_TASK_CONFIGS["sim_stack_cube_scripted"]["num_episodes"], 100)
        self.assertEqual(SIM_TASK_CONFIGS["sim_stack_cube_scripted"]["episode_len"], 260)

    def test_sample_stack_pose_shape(self):
        from utils import sample_stack_pose

        pose = sample_stack_pose()
        self.assertEqual(pose.shape, (14,))

    def test_sim_env_reset_contract(self):
        from sim_env import make_sim_env

        env = make_sim_env("sim_stack_cube_scripted")
        ts = env.reset()
        self.assertEqual(ts.observation["qpos"].shape, (14,))
        self.assertEqual(ts.observation["env_state"].shape, (14,))

    def test_ee_env_reset_contract(self):
        from ee_sim_env import make_ee_sim_env

        env = make_ee_sim_env("sim_stack_cube_scripted")
        ts = env.reset()
        self.assertEqual(ts.observation["qpos"].shape, (14,))
        self.assertEqual(ts.observation["env_state"].shape, (14,))

    def test_stack_policy_action_shape(self):
        from ee_sim_env import make_ee_sim_env
        from scripted_policy import StackCubePolicy

        env = make_ee_sim_env("sim_stack_cube_scripted")
        ts = env.reset()
        policy = StackCubePolicy(False)
        action = policy(ts)
        self.assertEqual(action.shape, (16,))

    def test_sample_stack_pose_stays_in_safe_corridor(self):
        from utils import sample_stack_pose

        poses = np.stack([sample_stack_pose() for _ in range(128)])
        red = poses[:, :3]
        blue = poses[:, 7:10]

        self.assertTrue(np.all(red[:, 0] >= -0.24))
        self.assertTrue(np.all(red[:, 0] <= -0.12))
        self.assertTrue(np.all(blue[:, 0] >= 0.12))
        self.assertTrue(np.all(blue[:, 0] <= 0.24))
        self.assertTrue(np.all(np.abs(red[:, 1] - 0.5) <= 0.03))
        self.assertTrue(np.all(np.abs(blue[:, 1] - 0.5) <= 0.03))
        self.assertGreater(np.min(np.linalg.norm(red[:, :2] - blue[:, :2], axis=1)), 0.25)

    def test_gripper_changes_are_smoothed(self):
        from scripted_policy import StackCubePolicy

        policy = StackCubePolicy(False)
        value = policy._step_gripper(1.0, 0.0, 0.2)
        self.assertAlmostEqual(value, 0.8)

    def test_joint_trajectory_smoothing_limits_per_step_delta(self):
        from utils import smooth_joint_trajectory

        trajectory = np.array(
            [
                np.zeros(14),
                np.array([0.4] * 6 + [0.3] + [-0.4] * 6 + [-0.3]),
                np.array([0.9] * 6 + [0.7] + [-0.9] * 6 + [-0.7]),
            ],
            dtype=np.float64,
        )

        smoothed = smooth_joint_trajectory(
            trajectory,
            arm_delta_limit=0.08,
            gripper_delta_limit=0.05,
        )
        deltas = np.abs(np.diff(smoothed, axis=0))
        arm_indices = list(range(6)) + list(range(7, 13))
        gripper_indices = [6, 13]

        self.assertTrue(np.all(deltas[:, arm_indices] <= 0.080001))
        self.assertTrue(np.all(deltas[:, gripper_indices] <= 0.050001))

    def test_left_lift_failure_retries_instead_of_advancing(self):
        from scripted_policy import StackCubePolicy

        policy = StackCubePolicy(False)
        policy.left_phase = "lift_red"
        policy.left_phase_steps = 50
        policy.left_pick_quat = np.array([1.0, 0.0, 0.0, 0.0])
        obs = {"mocap_pose_left": np.array([0.0, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0])}
        red_pose = np.array([-0.16, 0.5, 0.02, 1.0, 0.0, 0.0, 0.0])

        policy._compute_left_target(obs, red_pose)
        self.assertEqual(policy.left_phase, "approach_red")

    def test_right_lift_failure_retries_instead_of_advancing(self):
        from scripted_policy import StackCubePolicy

        policy = StackCubePolicy(False)
        policy.right_phase = "lift_blue"
        policy.right_phase_steps = 50
        policy.right_pick_quat = np.array([1.0, 0.0, 0.0, 0.0])
        obs = {"mocap_pose_right": np.array([0.0, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0])}
        red_pose = np.array([0.0, 0.5, 0.02, 1.0, 0.0, 0.0, 0.0])
        blue_pose = np.array([0.16, 0.5, 0.02, 1.0, 0.0, 0.0, 0.0])

        policy._compute_right_target(obs, red_pose, blue_pose)
        self.assertEqual(policy.right_phase, "approach_blue")


if __name__ == "__main__":
    unittest.main()
