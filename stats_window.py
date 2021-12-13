import time

import main
import periscope.periscope as periscope
import pygame
from pygame._sdl2 import Window, Renderer, Texture, Image
import numpy as np
import sys

CAMZ_LOGO = "./camz.png"

test_mode = False

window = None
renderer = None

predicted_action_reward_graph = None
mean_episode_reward_graph = None
epsilon_graph = None
green_graph = None
red_graph = None
wall_graph = None

episode_text = None
win_text = None
lose_text = None
winrate_text = None
mean_text = None
zoom_text = None
offset_text = None
modifier_text = None

layout = None
layout_2 = None
layout_3 = None
layout_h = None
zoom_h = None
btn_h = None
layout_texture = None
game_texture = None
btn_texture = None
logo_texture = None

column_width = 300
font_size = 15

game_pos = [621, 301]
game_size = [256, 256]
border_size = 4
btn_pos = [game_pos[0], game_pos[1] + game_size[1] + border_size]
btn_size = [game_size[0], game_pos[1] / 10]

btn_fast_forward = None
btn_draw_lines = None

graphic_modifier = 1
paused = False
border_color = pygame.Color(0x000000ff)

class History:
    def __init__(self, size, show, interactive=False):
        self.buffer = np.zeros(size)
        self.size = size
        self.index = 0
        self.graphic_index = 0
        self.interactive = interactive
        if not interactive:
            self.show = size
        else:
            self.show = show
        self.zoom = 0

    def add(self, value):
        if self.index == self.size:
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[self.size - 1] = value
        else:
            self.buffer[self.index] = value
            self.index += 1
            self.graphic_index = max(0, self.index - self.show)
            self.zoom = 0

    def has_content(self):
        return self.index > 1

    def get_content(self):
        start_index = self.graphic_index + self.zoom
        end_index = min(start_index + self.show - self.zoom, self.index)
        if not self.interactive:
            start_index = 0
            end_index = self.index
        x = np.indices((1, end_index - start_index))[1][0]
        y = np.zeros(len(x))

        for i in range(start_index, end_index):
            y[i - start_index] = self.buffer[i]

        return x, y


class Counter:
    def __init__(self, string, counter=0):
        self.root = string
        self.counter = counter

    def increment(self):
        self.counter += 1

    def get_content(self):
        return self.root + str(self.counter)


class Value:
    def __init__(self, string, value):
        self.string = string
        self.value = value

    def update_content(self, value):
        self.value = value

    def get_content(self):
        return self.string + str(self.value)


class Mean:
    def __init__(self, string, size):
        self.string = string
        self.size = size
        self.buffer = []
        self.mean = 0

    def add(self, value):
        if len(self.buffer) == self.size:
            self.buffer.pop(0)
            self.buffer.append(value)
        else:
            self.buffer.append(value)
        self.mean = sum(self.buffer) / len(self.buffer)

    def get_content(self):
        return self.string + f'{self.mean:.2f}'


predicted_action_reward_history = History(333, 333)
mean_episode_reward_history = History(main.MAX_EPISODES, 100, True)
epsilon_history = History(10000, 10000)
green_history = History(main.MAX_EPISODES, 100, True)
red_history = History(main.MAX_EPISODES, 100, True)
wall_history = History(main.MAX_EPISODES, 100, True)
episode_counter = Counter("Episode ")
win_counter = Counter("Wins: ")
lose_counter = Counter("Loses: ")
winrate_mean = Mean("Winrate: ", 5000)
mean_reward_mean = Mean("Mean reward: ", 5000)
gamma_value = Value("Gamma: ", main.gamma)
greedy_frames_value = Value("Greedy frames: ", main.epsilon_greedy_frames)
max_mem_len_value = Value("Max memory length: ", main.max_memory_length)
update_target_network_value = Value("Update target network: ", main.update_target_network)
batch_size_value = Value("Batch size: ", main.batch_size)


def activate_btn(btn, swap, obj):
    if btn.color_background == pygame.Color(0xffcc33ff):
        btn.color_font = pygame.Color(0xffcc33ff)
        btn.color_background = pygame.Color(0x1e2320ff)
    else:
        btn.color_background = pygame.Color(0xffcc33ff)
        btn.color_font = pygame.Color(0x1e2320ff)
    btn.update()
    if swap == "ff":
        obj.force_fps = not obj.force_fps
    else:
        obj.DRAW_DISTANCES = not obj.DRAW_DISTANCES


def game_paused(g_p):
    global border_color, paused
    paused = g_p
    if g_p:
        border_color = pygame.Color(0xffcc33ff)
    else:
        border_color = pygame.Color(0x000000ff)


def save_history(file, history):
    for i in range(history.index):
        file.write(f'{history.buffer[i]} ')

    file.write('\n')


def save_counter(file, counter):
    file.write(f'{counter.counter}\n')


def save_mean(file, mean):
    for i in range(len(mean.buffer)):
        file.write(f'{mean.buffer[i]} ')
    file.write('\n')


def save(file):
    global predicted_action_reward_history, mean_episode_reward_history, epsilon_history, green_history, red_history, wall_history, \
    episode_counter, win_counter, lose_counter, winrate_mean, mean_reward_mean

    save_history(file, predicted_action_reward_history)
    save_history(file, mean_episode_reward_history)
    save_history(file, epsilon_history)
    save_history(file, green_history)
    save_history(file, red_history)
    save_history(file, wall_history)

    save_counter(file, episode_counter)
    save_counter(file, win_counter)
    save_counter(file, lose_counter)

    save_mean(file, winrate_mean)
    save_mean(file, mean_reward_mean)


def load_history(file, history):
    str = file.readline().split(' ')

    for i in range(len(str)):
        if str[i].isspace():
            break
        history.add(float(str[i]))


def load_counter(file, counter):
    counter.counter = int(file.readline())


def load_mean(file, mean):
    str = file.readline().split(' ')

    for i in range(len(str)):
        if str[i].isspace():
            break
        mean.add(float(str[i]))


def load(file):
    global predicted_action_reward_history, mean_episode_reward_history, epsilon_history, green_history, red_history, wall_history, \
        episode_counter, win_counter, lose_counter, winrate_mean, mean_reward_mean

    load_history(file, predicted_action_reward_history)
    load_history(file, mean_episode_reward_history)
    load_history(file, epsilon_history)
    load_history(file, green_history)
    load_history(file, red_history)
    load_history(file, wall_history)

    load_counter(file, episode_counter)
    load_counter(file, win_counter)
    load_counter(file, lose_counter)

    load_mean(file, winrate_mean)
    load_mean(file, mean_reward_mean)


def init(is_test_mode):
    global window, renderer, predicted_action_reward_graph, mean_episode_reward_graph, epsilon_graph, layout, layout_texture, \
        wall_graph, red_history, green_history, wall_history, episode_counter, win_counter, lose_counter, winrate_mean, mean_reward_mean, \
        gamma_value, greedy_frames_value, max_mem_len_value, update_target_network_value, batch_size_value, layout_2, layout_3, layout_h, \
        game_texture, btn_h, btn_draw_lines, btn_fast_forward, btn_texture, update_lines_func, update_ff_func, green_graph, red_graph, \
        wall_graph, episode_text, win_text, lose_text, winrate_text, mean_text, zoom_text, offset_text, modifier_text, zoom_h, test_mode, logo_texture

    test_mode = is_test_mode
    pygame.init()

    window = Window("stats", resizable=False, size=(column_width * 3, 600))
    renderer = Renderer(window)

    predicted_action_reward_graph = periscope.LinePlot(column_width, 150)
    mean_episode_reward_graph = periscope.LinePlot(column_width, 150)
    epsilon_graph = periscope.LinePlot(column_width, 150)
    green_graph = periscope.LinePlot(column_width, 150, y_ints=True)
    red_graph = periscope.LinePlot(column_width, 150, y_ints=True)
    wall_graph = periscope.LinePlot(column_width, 150, y_ints=True)

    btn_fast_forward = periscope.TextField(int(game_size[0] / 2 - border_size * 1.5), "Fast forward",
                                           font_size=font_size,
                                           font_name="consolas", color_font=pygame.Color(0xffcc33ff),
                                           color_background=pygame.Color(0x1e2320ff), align="middle")
    btn_draw_lines = periscope.TextField(int(game_size[0] / 2 - border_size * 1.5), "Draw lines", font_size=font_size,
                                         font_name="consolas", color_font=pygame.Color(0xffcc33ff),
                                         color_background=pygame.Color(0x1e2320ff), align="middle")

    btn_draw_lines.padding = 0
    btn_fast_forward.padding = 0

    if not test_mode:
        layout = periscope.VStack([
            periscope.TextField(column_width, "Predicted action reward", font_size=font_size, font_name="consolas"),
            predicted_action_reward_graph,
            periscope.TextField(column_width, "Mean episode reward", font_size=font_size, font_name="consolas"),
            mean_episode_reward_graph,
            periscope.TextField(column_width, "Epsilon", font_size=font_size, font_name="consolas"),
            epsilon_graph
        ])
    else:
        layout = periscope.VStack([
            periscope.TextField(column_width, "Predicted action reward", font_size=font_size, font_name="consolas"),
            predicted_action_reward_graph,
            periscope.TextField(column_width, "Mean episode reward", font_size=font_size, font_name="consolas"),
            mean_episode_reward_graph,
            periscope.TextField(column_width, "Made by", font_size=font_size, font_name="consolas"),
        ])

    layout_2 = periscope.VStack([
        periscope.TextField(column_width, "Green blobs eaten", font_size=font_size, font_name="consolas"),
        green_graph,
        periscope.TextField(column_width, "Red blobs eaten", font_size=font_size, font_name="consolas"),
        red_graph,
        periscope.TextField(column_width, "Frames spent near walls", font_size=font_size, font_name="consolas"),
        wall_graph,
    ])

    episode_text = periscope.TextField(column_width, episode_counter.get_content(), font_size=font_size,
                                       font_name="consolas")
    win_text = periscope.TextField(column_width, win_counter.get_content(), font_size=font_size, font_name="consolas")
    lose_text = periscope.TextField(column_width, lose_counter.get_content(), font_size=font_size, font_name="consolas")
    winrate_text = periscope.TextField(column_width, winrate_mean.get_content(), font_size=font_size,
                                       font_name="consolas")
    mean_text = periscope.TextField(column_width, mean_reward_mean.get_content(), font_size=font_size,
                                    font_name="consolas")

    zoom_text = periscope.TextField(int(column_width / 3), "Zoom: 0", font_size=font_size - 2, font_name="consolas",
                                    color_font=pygame.Color(0xffcc33ff))
    offset_text = periscope.TextField(int(column_width / 3), "Offset: 0", font_size=font_size - 2, font_name="consolas",
                                    color_font=pygame.Color(0xffcc33ff))
    modifier_text = periscope.TextField(int(column_width / 3), "Modifier: 1", font_size=font_size - 2, font_name="consolas",
                                    color_font=pygame.Color(0xffcc33ff))

    zoom_h = periscope.HStack([
        zoom_text, offset_text, modifier_text
    ], border_size=0)

    layout_3 = periscope.VStack([
        periscope.TextField(column_width, "Stats", font_size=font_size, font_name="consolas",
                            color_font=pygame.Color(0xffcc33ff)),
        episode_text,
        win_text,
        lose_text,
        winrate_text,
        mean_text,
        periscope.TextField(column_width, "Parameters", font_size=font_size, font_name="consolas",
                            color_font=pygame.Color(0xffcc33ff)),
        periscope.TextField(column_width, gamma_value.get_content(), font_size=font_size, font_name="consolas"),
        periscope.TextField(column_width, greedy_frames_value.get_content(), font_size=font_size, font_name="consolas"),
        periscope.TextField(column_width, max_mem_len_value.get_content(), font_size=font_size, font_name="consolas"),
        periscope.TextField(column_width, update_target_network_value.get_content(), font_size=font_size,
                            font_name="consolas"),
        periscope.TextField(column_width, batch_size_value.get_content(), font_size=font_size, font_name="consolas"),
        zoom_h
    ])

    layout_h = periscope.HStack([
        layout, layout_2, layout_3
    ])

    btn_h = periscope.HStack([
        btn_draw_lines, btn_fast_forward
    ], border_size=1, color_border=pygame.Color(0xffcc33ff))

    layout_texture = Texture(renderer, (layout_h.w, layout_h.h), streaming=True)

    game_texture = Texture(renderer, (game_size[0], game_size[1]), streaming=True)

    btn_texture = Texture(renderer, (btn_h.w, btn_h.h), streaming=True)

    if test_mode:
        logo_texture = Image(Texture.from_surface(renderer, pygame.image.load(CAMZ_LOGO)))


click_pressed = False
key_pressed = False


def move_index(change):
    global mean_episode_reward_history, green_history, red_history, wall_history, paused
    if not paused:
        return
    mean_episode_reward_history.graphic_index = min(max(-mean_episode_reward_history.zoom, mean_episode_reward_history.graphic_index + change),
                                                    max(0, mean_episode_reward_history.index -
                                                        mean_episode_reward_history.show))
    green_history.graphic_index = min(max(-green_history.zoom, green_history.graphic_index + change),
                                      max(0, green_history.index - green_history.show))
    red_history.graphic_index = min(max(-red_history.zoom, red_history.graphic_index + change),
                                    max(0, red_history.index - red_history.show))
    wall_history.graphic_index = min(max(-wall_history.zoom, wall_history.graphic_index + change),
                                     max(0, wall_history.index - wall_history.show))


def zoom_graphic(change):
    global mean_episode_reward_history, green_history, red_history, wall_history, paused
    if not paused:
        return
    mean_episode_reward_history.zoom = min(
        max(-mean_episode_reward_history.graphic_index, mean_episode_reward_history.zoom + change),
        mean_episode_reward_history.show - 2, mean_episode_reward_history.index - 2)
    green_history.zoom = min(max(-green_history.graphic_index, green_history.zoom + change), green_history.show - 2,
                             green_history.index - 2)
    red_history.zoom = min(max(-red_history.graphic_index, red_history.zoom + change), red_history.show - 2,
                           red_history.index - 2)
    wall_history.zoom = min(max(-wall_history.graphic_index, wall_history.zoom + change), wall_history.show - 2,
                            wall_history.index - 2)


def update(p, game):
    global window, renderer, predicted_action_reward_graph, mean_episode_reward_graph, epsilon_graph, layout, layout_texture, \
        wall_graph, red_history, green_history, wall_history, episode_counter, win_counter, lose_counter, winrate_mean, mean_reward_mean, \
        gamma_value, greedy_frames_value, max_mem_len_value, update_target_network_value, batch_size_value, layout_2, layout_3, layout_h, \
        game_texture, btn_h, btn_draw_lines, btn_fast_forward, btn_texture, update_lines_func, update_ff_func, click_pressed, green_graph, \
        red_graph, wall_graph, episode_text, win_text, lose_text, winrate_text, mean_text, graphic_modifier, key_pressed, zoom_text, \
        offset_text, modifier_text, zoom_h, border_color, logo_texture, test_mode

    if pygame.mouse.get_pressed()[0]:
        if click_pressed == False:
            pos = pygame.mouse.get_pos()
            click_pressed = True
            if pos[0] < btn_pos[0] + btn_size[0] / 2 and pos[0] > btn_pos[0] and pos[1] < btn_pos[1] + btn_size[1] and \
                    pos[1] > btn_pos[1]:
                activate_btn(btn_draw_lines, "lines", game)
            elif pos[0] < btn_pos[0] + btn_size[0] and pos[0] > btn_pos[0] + btn_size[0] / 2 and pos[1] < btn_pos[1] + \
                    btn_size[1] and pos[1] > btn_pos[1]:
                activate_btn(btn_fast_forward, "ff", p)
    else:
        click_pressed = False

    if pygame.key.get_pressed()[pygame.constants.K_LEFT]:  # go k episodes back
        if not key_pressed:
            key_pressed = True
            move_index(-graphic_modifier)
    elif pygame.key.get_pressed()[pygame.constants.K_RIGHT]:  # go k episodes forward
        if not key_pressed:
            key_pressed = True
            move_index(graphic_modifier)
    elif pygame.key.get_pressed()[pygame.constants.K_UP]:  # zoom out
        if not key_pressed:
            key_pressed = True
            zoom_graphic(-graphic_modifier)
    elif pygame.key.get_pressed()[pygame.constants.K_DOWN]:  # zoom in
        if not key_pressed:
            key_pressed = True
            zoom_graphic(graphic_modifier)
    elif pygame.key.get_pressed()[pygame.constants.K_KP_PLUS]:
        if not key_pressed:
            key_pressed = True
            graphic_modifier = min(100, graphic_modifier + 1)
            # print(f"Zoom modifier is now {graphic_modifier}", file=sys.stderr)
    elif pygame.key.get_pressed()[pygame.constants.K_KP_MINUS]:
        if not key_pressed:
            key_pressed = True
            graphic_modifier = max(1, graphic_modifier - 1)
            # print(f"Zoom modifier is now {graphic_modifier}", file=sys.stderr)
    else:
        key_pressed = False

    offset_text.set_content(f'Offset: {max(0, red_history.index - red_history.graphic_index - red_history.show)}')
    zoom_text.set_content(f'Zoom: {red_history.zoom}')
    modifier_text.set_content(f'Modifier: {graphic_modifier}')

    renderer.clear()
    renderer.draw_color = (0x1e, 0x23, 0x20, 0xff)
    renderer.fill_rect(pygame.Rect(0, 0, column_width * 3, 600))

    if predicted_action_reward_history.has_content():
        pred_x, pred_y = predicted_action_reward_history.get_content()
        predicted_action_reward_graph.set_content(pred_x, pred_y)

    if mean_episode_reward_history.has_content():
        mean_x, mean_y = mean_episode_reward_history.get_content()
        mean_episode_reward_graph.set_content(mean_x, mean_y)

    if epsilon_history.has_content():
        eps_x, eps_y = epsilon_history.get_content()
        epsilon_graph.set_content(eps_x, eps_y)

    if green_history.has_content():
        green_x, green_y = green_history.get_content()
        green_graph.set_content(green_x, green_y)

    if red_history.has_content():
        red_x, red_y = red_history.get_content()
        red_graph.set_content(red_x, red_y)

    if wall_history.has_content():
        wall_x, wall_y = wall_history.get_content()
        wall_graph.set_content(wall_x, wall_y)

    episode_text.set_content(episode_counter.get_content())
    win_text.set_content(win_counter.get_content())
    lose_text.set_content(lose_counter.get_content())
    winrate_text.set_content(winrate_mean.get_content())
    mean_text.set_content(mean_reward_mean.get_content())

    layout_texture.update(layout_h.surface)
    layout_texture.draw(dstrect=pygame.Rect(0, 0, column_width * 3, 600))
    if test_mode:
        logo_texture.draw(dstrect=(4, 422, 293, 173))

    game_texture.update(pygame.display.get_surface())
    game_texture.draw(dstrect=pygame.Rect(game_pos[0], game_pos[1], game_size[0], game_size[1]))

    for i in range(1, border_size + 1):
        renderer.draw_color = border_color
        renderer.draw_rect((game_pos[0] - i, game_pos[1] - i, game_size[0] + i * 2, game_size[1] + i * 2))
        renderer.draw_color = (0x1e, 0x23, 0x20, 0xff)

    btn_texture.update(btn_h.surface)
    btn_texture.draw(dstrect=pygame.Rect(btn_pos[0], btn_pos[1], btn_size[0], btn_size[1]))

    for i in range(1, border_size + 1):
        renderer.draw_color = border_color
        renderer.draw_rect((btn_pos[0] - i, btn_pos[1] - i, btn_size[0] + i * 2, btn_size[1] + i * 2))
        renderer.draw_color = (0x1e, 0x23, 0x20, 0xff)

    renderer.present()
