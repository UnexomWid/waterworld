import pygame
import sys
import math

# import .base
from .base.pygamewrapper import PyGameWrapper

from .utils.vec2d import vec2d
from .utils import percent_round_int
from pygame.constants import K_w, K_a, K_s, K_d
from .primitives import Player, Creep


class WaterWorld(PyGameWrapper):
    """
    Based Karpthy's WaterWorld in `REINFORCEjs`_.

    .. _REINFORCEjs: https://github.com/karpathy/reinforcejs

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    num_creeps : int (default: 3)
        The number of creeps on the screen at once.
    """

    def __init__(self,
                 width=256,
                 height=256,
                 num_creeps=3, draw_screen=True):
        self.draw_screen = draw_screen
        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.BG_COLOR = (255, 255, 255)
        self.N_CREEPS = num_creeps
        self.CREEP_TYPES = ["GOOD", "BAD"]
        self.CREEP_COLORS = [(40, 140, 40), (150, 95, 95)]
        radius = percent_round_int(width, 0.047)
        self.CREEP_RADII = [radius, radius]
        self.CREEP_REWARD = [
            self.rewards["positive"],
            self.rewards["negative"]]
        self.CREEP_SPEED = 0.25 * width
        self.AGENT_COLOR = (60, 60, 140)
        self.AGENT_SPEED = 0.25 * width
        self.AGENT_RADIUS = radius
        self.AGENT_INIT_POS = (self.width / 2, self.height / 2)
        self.DRAW_DISTANCES = False
        self.creep_counts = {
            "GOOD": 0,
            "BAD": 0
        }

        self.dx = 0
        self.dy = 0
        self.player = None
        self.creeps = None

    def _handle_player_events(self):
        self.dx = 0
        self.dy = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    self.dx -= self.AGENT_SPEED

                if key == self.actions["right"]:
                    self.dx += self.AGENT_SPEED

                if key == self.actions["up"]:
                    self.dy -= self.AGENT_SPEED

                if key == self.actions["down"]:
                    self.dy += self.AGENT_SPEED

    def _add_creep(self):
        creep_type = self.rng.choice([0, 1])

        creep = None
        pos = (0, 0)
        dist = 0.0
        radius = self.CREEP_RADII[creep_type] * 2.5
        while dist < radius * 4 / 2.5:
            pos = self.rng.uniform(radius, self.height - radius, size=2)
            dist = math.sqrt(
                (self.player.pos.x - pos[0]) ** 2 + (self.player.pos.y - pos[1]) ** 2)

        # Modified because quite often creeps would spawn into the player. That's just unfair
        # dist = 0.0
        #
        # while dist < 1.5:
        #     radius = self.CREEP_RADII[creep_type] * 1.5
        #     pos = self.rng.uniform(radius, self.height - radius, size=2)
        #     dist = math.sqrt(
        #         (self.player.pos.x - pos[0]) ** 2 + (self.player.pos.y - pos[1]) ** 2)

        creep = Creep(
            self.CREEP_COLORS[creep_type],
            self.CREEP_RADII[creep_type],
            pos,
            self.rng.choice([-1, 1], 2),
            self.rng.rand() * self.CREEP_SPEED,
            self.CREEP_REWARD[creep_type],
            self.CREEP_TYPES[creep_type],
            self.width,
            self.height,
            self.rng.rand()
        )

        self.creeps.add(creep)

        self.creep_counts[self.CREEP_TYPES[creep_type]] += 1

    def getGameState(self):
        """

        Returns
        -------

        dict
            * player x position.
            * player y position.
            * player x velocity.
            * player y velocity.
            * player distance to each creep


        """

        min_bad, min_bad_2, min_good = self._get_nearest_creeps()
        # surround_good = self._get_num_surrounding_creeps(min_good, self.player.radius * 4)

        return [
            (self.player.pos.x + self.AGENT_RADIUS) / self.width,
            (self.player.pos.y + self.AGENT_RADIUS) / self.height,
            self.player.vel.x / self.width,
            self.player.vel.y / self.height,
            min_bad["relative_x"] / self.width,
            min_bad["relative_y"] / self.height,
            min_bad_2["relative_x"] / self.width,
            min_bad_2["relative_y"] / self.height,
            min_good["relative_x"] / self.width,
            min_good["relative_y"] / self.height
        ]

        # state = {
        #     "player_x": self.player.pos.x + self.AGENT_RADIUS,
        #     "player_y": self.player.pos.y + self.AGENT_RADIUS,
        #     "player_velocity_x": self.player.vel.x,
        #     "player_velocity_y": self.player.vel.y,
        #     "creep_dist": {
        #         "GOOD": [],
        #         "BAD": []
        #     },
        #     "creep_pos": {
        #         "GOOD": [],
        #         "BAD": []
        #     }
        # }
        #
        # for c in self.creeps:
        #     dist = math.sqrt((self.player.pos.x - c.pos.x) **
        #                      2 + (self.player.pos.y - c.pos.y)**2)
        #     state["creep_dist"][c.TYPE].append(dist)
        #     state["creep_pos"][c.TYPE].append([c.pos.x, c.pos.y])

        return state

    def getScore(self):
        return self.score

    def game_over(self):
        """
            Return bool if the game has 'finished'
        """
        return (self.creep_counts['GOOD'] == 0)

    def init(self):
        """
            Starts/Resets the game to its inital state
        """
        self.creep_counts = {"GOOD": 0, "BAD": 0}

        if self.player is None:
            self.player = Player(
                self.AGENT_RADIUS, self.AGENT_COLOR,
                self.AGENT_SPEED, self.AGENT_INIT_POS,
                self.width, self.height
            )

        else:
            self.player.pos = vec2d(self.AGENT_INIT_POS)
            self.player.vel = vec2d((0.0, 0.0))

        if self.creeps is None:
            self.creeps = pygame.sprite.Group()
        else:
            self.creeps.empty()

        for i in range(self.N_CREEPS):
            self._add_creep()

        self.score = 0
        self.ticks = 0
        self.lives = -1

    def step(self, dt):
        """
            Perform one step of game emulation.
        """
        dt /= 1000.0
        self.screen.fill(self.BG_COLOR)

        self.score += self.rewards["tick"]

        self._handle_player_events()
        self.player.update(self.dx, self.dy, dt)

        hits = pygame.sprite.spritecollide(self.player, self.creeps, True, collided=circle_collision)
        for creep in hits:
            self.creep_counts[creep.TYPE] -= 1
            self.score += creep.reward
            self._add_creep()

        if self.creep_counts["GOOD"] == 0:
            self.score += self.rewards["win"]

        self.creeps.update(dt)

        self.player.draw(self.screen)
        self.creeps.draw(self.screen)

    # def _get_num_surrounding_creeps(self, creep, radius):
    #     x = creep["relative_x"]
    #     y = creep["relative_y"]
    #
    #     num = 0
    #
    #     for c in self.creeps:
    #         if c.TYPE == "BAD":
    #             dist = math.sqrt((x - c.pos.x) ** 2 + (y - c.pos.y) ** 2)
    #             if dist <= radius:
    #                 num += 1
    #
    #     return num

    def _get_nearest_creeps(self):
        player_x = self.player.pos.x + self.AGENT_RADIUS
        player_y = self.player.pos.y + self.AGENT_RADIUS

        min_bad = {
            "x": player_x,
            "y": player_y,
            "relative_x": 0.0,
            "relative_y": 0.0,
            "distance": -1
        }

        min_bad_2 = {
            "x": player_x,
            "y": player_y,
            "relative_x": 0.0,
            "relative_y": 0.0,
            "distance": -1
        }

        min_good = {
            "x": player_x,
            "y": player_y,
            "relative_x": 0.0,
            "relative_y": 0.0,
            "distance": -1
        }

        for c in self.creeps:
            dist = math.sqrt((player_x - c.pos.x) **
                             2 + (player_y - c.pos.y) ** 2)

            if c.TYPE == "BAD":
                if min_bad_2["distance"] == -1:
                    min_bad_2["x"] = c.pos.x
                    min_bad_2["y"] = c.pos.y
                    min_bad_2["relative_x"] = c.pos.x - player_x
                    min_bad_2["relative_y"] = c.pos.y - player_y
                    min_bad_2["distance"] = dist
                if min_bad["distance"] == -1 or min_bad["distance"] > dist:
                    min_bad_2["x"] = min_bad["x"]
                    min_bad_2["y"] = min_bad["y"]
                    min_bad_2["relative_x"] = min_bad["relative_x"]
                    min_bad_2["relative_y"] = min_bad["relative_y"]
                    min_bad_2["distance"] = min_bad["distance"]
                    min_bad["x"] = c.pos.x
                    min_bad["y"] = c.pos.y
                    min_bad["relative_x"] = c.pos.x - player_x
                    min_bad["relative_y"] = c.pos.y - player_y
                    min_bad["distance"] = dist
                elif min_bad_2["distance"] > dist:
                    min_bad_2["x"] = c.pos.x
                    min_bad_2["y"] = c.pos.y
                    min_bad_2["relative_x"] = c.pos.x - player_x
                    min_bad_2["relative_y"] = c.pos.y - player_y
                    min_bad_2["distance"] = dist
            elif c.TYPE == "GOOD":
                if min_good["distance"] == -1 or min_good["distance"] > dist:
                    min_good["x"] = c.pos.x
                    min_good["y"] = c.pos.y
                    min_good["relative_x"] = c.pos.x - player_x
                    min_good["relative_y"] = c.pos.y - player_y
                    min_good["distance"] = dist

        return (min_bad, min_bad_2, min_good)

    def _draw_frame(self, draw_screen):
        """
        Decides if the screen will be drawn too
        """

        if self.draw_screen:
            if self.DRAW_DISTANCES:
                player_x = self.player.pos.x + self.AGENT_RADIUS
                player_y = self.player.pos.y + self.AGENT_RADIUS

                min_bad, min_bad_2, min_good = self._get_nearest_creeps()

                pygame.draw.line(pygame.display.get_surface(), (255, 0, 0), (player_x, player_y),
                                 (min_bad["x"], min_bad["y"]), width=3)
                pygame.draw.line(pygame.display.get_surface(), (255, 0, 0), (player_x, player_y),
                                 (min_bad_2["x"], min_bad_2["y"]), width=3)
                pygame.draw.line(pygame.display.get_surface(), (0, 255, 0), (player_x, player_y),
                                 (min_good["x"], min_good["y"]), width=3)

            pygame.display.update()


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = WaterWorld(width=256, height=256, num_creeps=10)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()


# Added collision for circles, since the game used collision for rectangles, while the creeps and the player were
# circles :|
def circle_collision(entity_one, entity_two):
    dist = math.sqrt((entity_one.pos.x - entity_two.pos.x + entity_two.radius) ** 2 + (
                entity_one.pos.y - entity_two.pos.y + entity_two.radius) ** 2)
    return dist <= entity_one.radius + entity_two.radius