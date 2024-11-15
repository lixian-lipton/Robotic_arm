import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization算法（PPO）（截断版本）

    参考文献：
    - 论文：https://arxiv.org/abs/1707.06347
    - 代码：此实现借鉴了OpenAI Spinning Up（https://github.com/openai/spinningup/）
    - 其他资源：https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: 使用的策略模型（MlpPolicy, CnnPolicy, ...）
    :param env: 要学习的环境（如果在Gym中注册，可以是字符串）
    :param learning_rate: 学习率，可以是当前进度的函数（从1到0）
    :param n_steps: 每次更新的环境运行步数
    :param batch_size: 小批量大小
    :param n_epochs: 优化代理损失的迭代次数
    :param gamma: 折扣因子
    :param gae_lambda: 一般化优势估计的偏差与方差的权衡因子
    :param clip_range: 策略更新的截断参数
    :param clip_range_vf: 价值函数的截断参数
    :param normalize_advantage: 是否对优势进行标准化
    :param ent_coef: 用于损失计算的熵系数
    :param vf_coef: 用于损失计算的价值函数系数
    :param max_grad_norm: 梯度裁剪的最大值
    :param use_sde: 是否使用广义状态依赖探索（gSDE）
    :param sde_sample_freq: 使用gSDE时每n步采样一次噪声矩阵
    :param rollout_buffer_class: 使用的滚动缓冲区类
    :param rollout_buffer_kwargs: 创建时传递给滚动缓冲区的关键字参数
    :param target_kl: 更新之间KL散度的限制
    :param stats_window_size: 日志统计的窗口大小
    :param tensorboard_log: TensorBoard日志的位置
    :param policy_kwargs: 传递给策略的额外参数
    :param verbose: 输出的详细程度
    :param seed: 随机数生成器的种子
    :param device: 代码运行的设备（cpu, cuda）
    :param _init_setup_model: 是否在创建时构建网络
    """

    # 策略名称与其对应类的映射
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
            rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        # 初始化父类
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,  # 防止在此处设置模型
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # 优势标准化的合理性检查
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` 必须大于 1."

        if self.env is not None:
            # 检查 n_steps * n_envs > 1，以避免在优势标准化时出现 NaN
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` 必须大于 1. 当前 n_steps={self.n_steps} 和 n_envs={self.env.num_envs}"

            # 确保滚动缓冲区大小是小批量大小的整数倍
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"你指定的小批量大小为 {batch_size}，"
                    f"但因为`RolloutBuffer`的大小为`n_steps * n_envs = {buffer_size}`，"
                    f"在每 {untruncated_batches} 个未截断的小批量后，"
                    f"将会有一个大小为 {buffer_size % batch_size} 的截断小批量。\n"
                    f"我们建议使用一个 `batch_size`，使其为 `n_steps * n_envs` 的因子。\n"
                    f"信息: (n_steps={self.n_steps} 和 n_envs={self.env.num_envs})"
                )

        # 将参数赋值给实例变量
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        # 如果请求，则设置模型
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # 调用父类的设置方法
        super()._setup_model()

        # 初始化策略和价值函数的截断范围
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` 必须为正数。"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        使用当前收集的滚动缓冲区更新策略。
        """
        # 切换到训练模式（影响批归一化和丢弃）
        self.policy.set_training_mode(True)

        # 更新优化器的学习率
        self._update_learning_rate(self.policy.optimizer)

        # 根据进度计算当前的截断范围
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]

        # 可选：计算价值函数的截断范围
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # 存储损失的列表
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # 训练 n_epochs 次
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # 遍历滚动缓冲区的小批量
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # 将离散动作从浮点数转换为长整型
                    actions = rollout_data.actions.long().flatten()

                # 使用 gSDE 时重置策略的噪声
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # 评估当前策略在选定动作上的表现
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # 标准化优势
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 计算新旧策略的比率
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # 计算截断的代理损失
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # 记录策略损失和截断比例
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # 处理价值函数预测的截断
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    # 截断新旧价值预测的差异
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                # 使用 TD 目标计算价值损失
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # 计算熵损失以鼓励探索
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)  # 近似熵
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # 将损失合并用于优化
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # 计算近似KL散度以便于提前停止
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                # 根据KL散度检查提前停止条件
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"在第 {epoch} 步骤提前停止，因达到最大KL: {approx_kl_div:.2f}")
                    break

                # 优化步骤
                self.policy.optimizer.zero_grad()  # 清除之前的梯度
                loss.backward()  # 反向传播损失
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # 裁剪梯度
                self.policy.optimizer.step()  # 更新策略参数

            self._n_updates += 1  # 增加更新计数
            if not continue_training:  # 检查是否继续训练
                break

        # 计算解释方差以用于日志记录
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # 记录各种指标
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # 额外日志记录
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self: SelfPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfPPO:
        # 调用父类的学习方法开始训练
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )