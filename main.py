from ple.games.waterworld import WaterWorld
from ple import PLE
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pygame
from pygame.constants import K_f, K_SPACE, K_BACKQUOTE
import stats_window

SCREEN_SIZE = 256
INTERNAL_SCREEN_SIZE = 84 # If using RGB, the internal matrix can be downscaled to this size
FPS = 30
FRAMES = 1000
MAX_EPISODES = 3000
NUM_CREEPS = 5
NUM_INPUTS = 4 + 2 * 3 # posX, posY, velX, velY, 3 blobs with X, Y
NUM_ACTIONS = 4
GRAPHICS = True

gamma = 0.99
epsilon = 0.5
epsilon_min = 0.0
epsilon_max = 1.0
epsilon_delta = (epsilon_max - epsilon_min)
batch_size = 32

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []

running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 5000
epsilon_greedy_frames = 50000.0
max_memory_length = 1000000
model_train_freq = 4
update_target_network = 1000
loss_function = keras.losses.Huber()

actions = None
model = None
target = None
optimizer = None
cost = None

paused = False

def create_model():
    inputs = layers.Input(shape=(NUM_INPUTS))

    layer1 = layers.Dense(100, activation="relu", kernel_regularizer="l2")(inputs)
    layer2 = layers.Dense(100, activation="relu", kernel_regularizer="l2")(layer1)
    layer3 = layers.Dense(10, activation="relu", kernel_regularizer="l2")(layer2)

    output = layers.Dense(NUM_ACTIONS, activation="linear")(layer3)

    return keras.Model(inputs=inputs, outputs=output)


def init(act):
    global actions, model, target, optimizer, cost
    actions = act

    model = create_model()
    target = create_model()

    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    cost = keras.losses.Huber()


def pick_action(act):
    global actions
    return actions[act]


def save(tag):
    global model, target, episode_count, frame_count, action_history, state_history, state_next_history,rewards_history, done_history, episode_reward_history, \
    running_reward, epsilon


    model.save(f'./data/{tag}/model')

    with open(f'./data/{tag}/checkpoint.dat', 'wt') as file:
        file.write(f'{episode_count}\n')
        file.write(f'{frame_count}\n')
        file.write(f'{running_reward}\n')
        file.write(f'{epsilon}\n')

        for i in range(len(action_history)):
            file.write(f'{action_history[i]} ')
        file.write('\n')

        for i in range(len(state_history)):
            for j in range(len(state_history[i])):
                file.write(f'{state_history[i][j]} ')
            file.write('\n')

        for i in range(len(state_next_history)):
            for j in range(len(state_next_history[i])):
                file.write(f'{state_next_history[i][j]} ')
            file.write('\n')

        for i in range(len(rewards_history)):
            file.write(f'{rewards_history[i]} ')
        file.write('\n')

        for i in range(len(done_history)):
            file.write(f'{ 1 if done_history[i] else 0} ')
        file.write('\n')

        for i in range(len(episode_reward_history)):
            file.write(f'{episode_reward_history[i]} ')
        file.write('\n')

        stats_window.save(file)

def load(tag):
    global model, target, episode_count, frame_count, action_history, state_history, state_next_history, rewards_history, done_history, episode_reward_history, \
    running_reward, epsilon

    model = tf.keras.models.load_model(f'./data/{tag}/model')
    target = tf.keras.models.load_model(f'./data/{tag}/model')

    with open(f'./data/{tag}/checkpoint.dat', 'rt') as file:
        episode_count = float(file.readline())
        frame_count = float(file.readline())
        running_reward = float(file.readline())
        epsilon = float(file.readline())

        str = file.readline().split(' ')

        for i in range(len(str)):
            if str[i].isspace():
                break
            action_history.append(int(str[i]))

        for i in range(len(action_history)):
            str = file.readline().split(' ')

            state_history.append([])

            for j in range(len(str)):
                if str[j].isspace():
                    break
                state_history[i].append(float(str[j]))

        for i in range(len(action_history)):
            str = file.readline().split(' ')

            state_next_history.append([])

            for j in range(len(str)):
                if str[j].isspace():
                    break
                state_next_history[i].append(float(str[j]))

        str = file.readline().split(' ')

        for i in range(len(str)):
            if str[i].isspace():
                break
            rewards_history.append(float(str[i]))

        str = file.readline().split(' ')

        for i in range(len(str)):
            if str[i].isspace():
                break
            done_history.append(True if int(str[i]) == 1 else 0)

        str = file.readline().split(' ')

        for i in range(len(str)):
            if str[i].isspace():
                break
            episode_reward_history.append(float())

        stats_window.load(file)


def train(tag=None):
    global actions, model, target, optimizer, cost, episode_count, SCREEN_SIZE, FPS, MAX_EPISODES, FRAMES, \
        epsilon, NUM_ACTIONS, epsilon_delta, epsilon_greedy_frames, epsilon_min, epsilon_max, \
        model_train_freq, batch_size, state_history, state_next_history, done_history, action_history, \
        episode_reward_history, rewards_history, paused, frame_count
    game = WaterWorld(width=SCREEN_SIZE, height=SCREEN_SIZE, num_creeps=8, draw_screen=True)
    p = PLE(game, fps=FPS, display_screen=True, force_fps=not GRAPHICS)
    p.init()
    init(p.getActionSet())
    rewards_history = []
    frame_count = 0

    f_pressed = False
    space_pressed = False
    grave_pressed = False

    if GRAPHICS:
        stats_window.init(False)

    if tag is not None:
        load(tag)

    i = episode_count

    while i < MAX_EPISODES:
        episode_reward = 0.0
        good_creeps = 0
        bad_creeps = 0
        wall_hugger = 0
        p.reset_game()
        state = p.getGameState()

        f = 0

        while f < FRAMES:
            keys = pygame.key.get_pressed()
            if keys[K_SPACE]:
                if not f_pressed:
                    f_pressed = True
                    paused = not paused
                    stats_window.game_paused(paused)
            else:
                f_pressed = False

            if keys[K_f]:
                if not space_pressed:
                    space_pressed = True
                    if not GRAPHICS:
                        p.force_fps = not p.force_fps
                    else:
                        stats_window.activate_btn(stats_window.btn_fast_forward, "ff", p)
            else:
                space_pressed = False

            if keys[K_BACKQUOTE]:
                if not grave_pressed:
                    grave_pressed = True
                    if not GRAPHICS:
                        game.DRAW_DISTANCES = not game.DRAW_DISTANCES
                    else:
                        stats_window.activate_btn(stats_window.btn_draw_lines, "lines", game)
                else:
                    grave_pressed = False

            if paused:
                if GRAPHICS:
                    stats_window.update(p, game)
                pygame.event.pump()
                continue

            if f % 4 != 0:
                time_elapsed = p._tick()
                p.game.step(time_elapsed)
                p._draw_frame()
                if GRAPHICS:
                    stats_window.update(p, game)
                f += 1
                continue

            frame_count += 1

            action_index = 0
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Random action if the game is in the first random frames
                action_index = np.random.choice(NUM_ACTIONS)
            else:
                # Predict Q-values
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_qval = model(state_tensor, training=False)

                action_index = tf.argmax(action_qval[0]).numpy()
                if GRAPHICS:
                    stats_window.predicted_action_reward_history.add(np.max(action_qval[0]))

            # 0.5 -> 0, 0, 0 -> 0.5, 0.5 -> 0, 0 -> 0.5, 0.5 -> 0, 0
            if frame_count < epsilon_greedy_frames:
                epsilon -= epsilon_delta / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)
            elif frame_count < 2 * epsilon_greedy_frames:
                epsilon = 0
            elif frame_count < 2.5 * epsilon_greedy_frames:
                epsilon += (epsilon_delta / epsilon_greedy_frames) / 2
                epsilon = min(epsilon, epsilon_max / 4)
            elif frame_count < 3 * epsilon_greedy_frames:
                epsilon -= (epsilon_delta / epsilon_greedy_frames) / 2
                epsilon = max(epsilon, epsilon_min)
            elif frame_count < 3.5 * epsilon_greedy_frames:
                epsilon += (epsilon_delta / epsilon_greedy_frames) / 2
                epsilon = min(epsilon, epsilon_max / 4)
            elif frame_count < 4 * epsilon_greedy_frames:
                epsilon -= (epsilon_delta / epsilon_greedy_frames) / 2
                epsilon = max(epsilon, epsilon_min)
            else:
                epsilon = 0

            if GRAPHICS:
                stats_window.epsilon_history.add(epsilon)

            action = pick_action(action_index)
            reward = p.act(action)

            if reward == 1:
                good_creeps += 1
            elif reward == -1:
                bad_creeps += 1

            if reward == -1:
                reward = -2.5

            # Touching walls is discouraged
            if (state[0] < 0.1 or state[0] > 0.9):
                reward -= 0.1
                wall_hugger += 1
            if (state[1] < 0.1 or state[1] > 0.9):
                reward -= 0.1
                wall_hugger += 1

            next_state = p.getGameState()
            done = p.game_over()

            next_state = np.array(next_state)

            episode_reward += reward

            action_history.append(action_index)
            state_history.append(state)
            state_next_history.append(next_state)
            done_history.append(done)
            rewards_history.append(reward)
            state = next_state

            if frame_count % model_train_freq == 0 and len(done_history) > batch_size:
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                future_rewards = target.predict(state_next_sample)
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

                # Final states have the value -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                masks = tf.one_hot(action_sample, NUM_ACTIONS)

                with tf.GradientTape() as tape:
                    q_values = model(state_sample)
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

                    loss = loss_function(updated_q_values, q_action)

                # Backpropagate
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                target.set_weights(model.get_weights())
                template = "Episode {}, Frame count {} ---> Ep. Mean frame reward = {:.2f} | Ep. Total reward = {} | Ep. Good creeps = {} | Ep. Bad creeps = {} | Won = {}"
                print(template.format(episode_count, frame_count, running_reward, episode_reward, good_creeps, bad_creeps, done))

                if frame_count % 10000 == 0:
                    save(str(frame_count))

            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]
            if GRAPHICS:
                stats_window.update(p, game)

            if done:
                break

            f += 1

        episode_reward_history.append(episode_reward)

        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]

        running_reward = np.mean(episode_reward_history)
        if GRAPHICS:
            stats_window.mean_episode_reward_history.add(running_reward)
            stats_window.green_history.add(good_creeps)
            stats_window.red_history.add(bad_creeps)
            stats_window.wall_history.add(wall_hugger)
            stats_window.episode_counter.increment()
            if done:
                stats_window.win_counter.increment()
            else:
                stats_window.lose_counter.increment()
            stats_window.winrate_mean.add(int(done))
            stats_window.mean_reward_mean.add(episode_reward)
        episode_count += 1
        if GRAPHICS:
            stats_window.update(p, game)

        i += 1

    save(frame_count)


def test(model_path, it=-100, blobs=5, log_path='log.txt'):
    global actions, model, target, optimizer, cost, episode_count, SCREEN_SIZE, FPS, MAX_EPISODES, FRAMES, \
        epsilon, NUM_ACTIONS, epsilon_delta, epsilon_greedy_frames, epsilon_min, epsilon_max, \
        model_train_freq, batch_size, state_history, state_next_history, done_history, action_history, \
        episode_reward_history, rewards_history, paused, frame_count, running_reward
    game = WaterWorld(width=SCREEN_SIZE, height=SCREEN_SIZE, num_creeps=blobs, draw_screen=True)
    p = PLE(game, fps=FPS, display_screen=True, force_fps=not GRAPHICS)
    p.init()
    init(p.getActionSet())
    model = tf.keras.models.load_model(model_path)
    target = model
    rewards_history = []
    frame_count = 0
    running_reward = 0
    episode_count = 0

    f_pressed = False
    space_pressed = False
    grave_pressed = False

    if GRAPHICS:
        stats_window.init(True)

    with open(log_path, 'w') as write_log:
        while it == -100 or episode_count < it:
            episode_reward = 0.0
            good_creeps = 0
            bad_creeps = 0
            wall_hugger = 0
            p.reset_game()
            state = p.getGameState()

            f = 0

            while f < FRAMES:
                keys = pygame.key.get_pressed()
                if keys[K_SPACE]:
                    if not f_pressed:
                        f_pressed = True
                        paused = not paused
                        stats_window.game_paused(paused)
                else:
                    f_pressed = False

                if keys[K_f]:
                    if not space_pressed:
                        space_pressed = True
                        if not GRAPHICS:
                            p.force_fps = not p.force_fps
                        else:
                            stats_window.activate_btn(stats_window.btn_fast_forward, "ff", p)
                else:
                    space_pressed = False

                if keys[K_BACKQUOTE]:
                    if not grave_pressed:
                        grave_pressed = True
                        if not GRAPHICS:
                            game.DRAW_DISTANCES = not game.DRAW_DISTANCES
                        else:
                            stats_window.activate_btn(stats_window.btn_draw_lines, "lines", game)
                else:
                    grave_pressed = False

                if paused:
                    if GRAPHICS:
                        stats_window.update(p, game)
                    pygame.event.pump()
                    continue

                if f % 4 != 0:
                    time_elapsed = p._tick()
                    p.game.step(time_elapsed)
                    p._draw_frame()
                    if GRAPHICS:
                        stats_window.update(p, game)
                    f += 1
                    continue

                frame_count += 1

                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_qvalues = model(state_tensor, training=False)

                action_index = tf.argmax(action_qvalues[0]).numpy()

                if GRAPHICS:
                    stats_window.predicted_action_reward_history.add(np.max(action_qvalues[0]))

                action = pick_action(action_index)
                reward = p.act(action)

                if reward == 1:
                    good_creeps += 1
                elif reward == -1:
                    bad_creeps += 1

                # Touching walls is discouraged
                if (state[0] < 0.1 or state[0] > 0.9):
                    wall_hugger += 1
                if (state[1] < 0.1 or state[1] > 0.9):
                    wall_hugger += 1

                next_state = p.getGameState()
                done = p.game_over()

                next_state = np.array(next_state)

                episode_reward += reward

                action_history.append(action_index)
                state_history.append(state)
                state_next_history.append(next_state)
                done_history.append(done)
                rewards_history.append(reward)
                state = next_state

                if frame_count % update_target_network == 0:
                    template = "Episode {}, Frame count {} ---> Ep. Mean frame reward = {:.2f} | Ep. Total reward = {} | Ep. Good creeps = {} | Ep. Bad creeps = {} | Won = {}"
                    print(template.format(episode_count, frame_count, running_reward, episode_reward, good_creeps, bad_creeps, done))

                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]
                if GRAPHICS:
                    stats_window.update(p, game)

                if done:
                    break

                f += 1

            episode_reward_history.append(episode_reward)

            if len(episode_reward_history) > 1000:
                del episode_reward_history[:1]

            running_reward = np.mean(episode_reward_history)

            if GRAPHICS:
                stats_window.mean_episode_reward_history.add(running_reward)
                stats_window.green_history.add(good_creeps)
                stats_window.red_history.add(bad_creeps)
                stats_window.wall_history.add(wall_hugger)
                stats_window.episode_counter.increment()
                if done:
                    stats_window.win_counter.increment()
                else:
                    stats_window.lose_counter.increment()
                stats_window.winrate_mean.add(int(done))
                stats_window.mean_reward_mean.add(episode_reward)

            # template = "Episode {}, Frame count {} ---> Ep. Mean frame reward = {:.2f} | Ep. Total reward = {} | Ep. Good creeps = {} | Ep. Bad creeps = {} | Won = {}\n"
            template = "{}:{}:{}:{}:{}:{}:{}\n"
            write_log.write(template.format(episode_count, frame_count, running_reward, episode_reward, good_creeps, bad_creeps,done))

            episode_count += 1

            if GRAPHICS:
                stats_window.update(p, game)

    save(frame_count)


def statistics():
    # Greedy
    test('best/data-greedy/437524/model', 1000, blobs=15, log_path='greedy_15.log')
    # Goes for the win
    test('best/data-wins/422809/model', 1000, blobs=15, log_path='win_15.log')
    # Cautious (best)
    test('best/data-cautious/437435/model', 1000, blobs=15, log_path='cautious_15.log')
    # Greedy
    test('best/data-greedy/437524/model', 1000, blobs=10, log_path='greedy_10.log')
    # Goes for the win
    test('best/data-wins/422809/model', 1000, blobs=10, log_path='win_10.log')
    # Cautious (best)
    test('best/data-cautious/437435/model', 1000, blobs=10, log_path='cautious_10.log')
    # Greedy
    test('best/data-greedy/437524/model', 1000, blobs=5, log_path='greedy_5.log')
    # Goes for the win
    test('best/data-wins/422809/model', 1000, blobs=5, log_path='win_5.log')
    # Cautious (best)
    test('best/data-cautious/437435/model', 1000, blobs=5, log_path='cautious_5.log')
    # Greedy
    test('best/data-greedy/437524/model', 1000, blobs=1, log_path='greedy_1.log')
    # Goes for the win
    test('best/data-wins/422809/model', 1000, blobs=1, log_path='win_1.log')
    # Cautious (best)
    test('best/data-cautious/437435/model', 1000, blobs=1, log_path='cautious_1.log')

if __name__ == '__main__':
    # train()
    # Greedy
    train()
    # test('best/data-greedy/437524/model', 1000, blobs=1)
    # Goes for the win
    # test('best/data-wins/422809/model', 1000, blobs=5)
    # Cautious (best)
    # test('best/data-cautious/437435/model', 1000, blobs=5)