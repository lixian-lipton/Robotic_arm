from env import Env
from team_algorithm import PPOAlgorithm, MyCustomAlgorithm
from time import sleep
import numpy as np
import self_solving
from robot_arm import Forward_solving

choose_model = 0
test = 1

def main(algorithm):
    env = Env(is_senior=False,seed=100,gui=test)
    done = False
    num_episodes = 100
    final_score = 0
    total_steps = 0
    total_distance = 0
    num_step = 0

    for i in range(num_episodes):
        score = 0
        done = False
        num_step = 0


        while not done:
            observation = env.get_observation()
            action = algorithm.get_action(observation)

            if test:
                observation = observation.flatten()
                num_step += 1
                print("Step:", num_step)
                # ini_robot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                # # self_gripper_position = self_solving.get_gripper_pos(ini_robot, observation)
                # self_gripper_position = Forward_solving(observation, action)
                # real_gripper_position = env.get_gripper_pos()
                # print("Self:", self_gripper_position, "Real:", real_gripper_position)
                sleep(0.2)

            obs = env.step(action)
            score += env.success_reward

            # Check if the episode has ended
            done = env.terminated

        total_steps += env.step_num
        total_distance += env.get_dis()
        final_score += score

        print(f"Test_{i} completed. steps:", env.step_num, "Distance:", env.get_dis(), "Score:", score)

    final_score /= num_episodes
    avg_distance = total_distance / num_episodes
    avg_steps = total_steps / num_episodes

    # After exiting the loop, get the total steps and final distance
    print("Test completed. Total steps:", avg_steps, "Final distance:", avg_distance, "Final score:", final_score)
    env.close()

if __name__ == "__main__":
    if choose_model == 0:
        algorithm = PPOAlgorithm()
    else:
        algorithm = MyCustomAlgorithm()
    main(algorithm)