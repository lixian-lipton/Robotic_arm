import unittest
import numpy as np
from model_train import MyRobotEnv, PPOTrainer

class TestMyRobotEnv(unittest.TestCase):

    def setUp(self):
        self.env = MyRobotEnv(gui=False)  # 创建环境实例，不启用 GUI
        self.env.reset()  # 重置环境以初始化状态

    def test_observation_space(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (1, 12), "Observation shape should be (1, 12)")

    def test_action_space(self):
        action = self.env.action_space.sample()  # 随机采样动作
        self.assertTrue(self.env.action_space.contains(action), "Sampled action should be within action space")

    def test_step_function(self):
        action = np.random.uniform(-1, 1, size=6)  # 生成随机动作
        obs, reward, done, _, _ = self.env.step(action)
        self.assertEqual(obs.shape, (1, 12), "Observation after step should be (1, 12)")
        self.assertIsInstance(reward, (int, float), "Reward should be a number")
        self.assertIn(done, [True, False], "Done should be a boolean value")

    def test_reset_function(self):
        obs, _ = self.env.reset()
        self.assertEqual(self.env.step_num, 0, "Step number should be reset to 0")
        self.assertEqual(obs.shape, (1, 12), "Observation after reset should be (1, 12)")

    def tearDown(self):
        self.env.close()  # 关闭环境

class TestPPOTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = PPOTrainer()

    def test_model_training(self):
        initial_model_path = "model.zip"
        self.trainer.model.save(initial_model_path)  # 保存初始模型以便后续比较
        self.trainer.train(total_timesteps=100)  # 训练模型
        self.trainer.save_model("test_model.zip")  # 保存测试模型

        # 确保模型训练后有变化（简单验证）
        test_model_path = "test_model.zip"
        self.assertNotEqual(initial_model_path, test_model_path, "Model should be updated after training")

    def tearDown(self):
        import os
        if os.path.exists("test_model.zip"):
            os.remove("test_model.zip")  # 清理测试生成的模型文件

if __name__ == "__main__":
    unittest.main()