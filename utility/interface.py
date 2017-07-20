import pygame, gym, numpy as np
from utility.utility import load_config, get_path
from utility.exp_replay import *

RGB_REPLAY_SHAPE = (4,
                    ((210, 160, 3), np.uint8),
                    ((), np.uint8),
                    ((), np.int32),
                    ((), np.bool))


class HumanInterface(object):
    def __init__(self, game=None, cfg='HI.yml'):
        self.cfg = load_config(cfg)
        self.game = game
        self._setup_env()
        self._setup_pygame()
        self.key_map = (
            0,  # 00000 none
            2,  # 00001 up
            5,  # 00010 down
            2,  # 00011 up/down (invalid)
            4,  # 00100 left
            7,  # 00101 up/left
            9,  # 00110 down/left
            7,  # 00111 up/down/left (invalid)
            3,  # 01000 right
            6,  # 01001 up/right
            8,  # 01010 down/right
            6,  # 01011 up/down/right (invalid)
            3,  # 01100 left/right (invalid)
            6,  # 01101 left/right/up (invalid)
            8,  # 01110 left/right/down (invalid)
            6,  # 01111 up/down/left/right (invalid)
            1,  # 10000 fire
            10,  # 10001 fire up
            13,  # 10010 fire down
            10,  # 10011 fire up/down (invalid)
            12,  # 10100 fire left
            15,  # 10101 fire up/left
            17,  # 10110 fire down/left
            15,  # 10111 fire up/down/left (invalid)
            11,  # 11000 fire right
            14,  # 11001 fire up/right
            16,  # 11010 fire down/right
            14,  # 11011 fire up/down/right (invalid)
            11,  # 11100 fire left/right (invalid)
            14,  # 11101 fire left/right/up (invalid)
            16,  # 11110 fire left/right/down (invalid)
            14  # 11111 fire up/down/left/right (invalid)
        )
        self.action_meaning = {
            0: "NOOP",
            1: "FIRE",
            2: "UP",
            3: "RIGHT",
            4: "LEFT",
            5: "DOWN",
            6: "UPRIGHT",
            7: "UPLEFT",
            8: "DOWNRIGHT",
            9: "DOWNLEFT",
            10: "UPFIRE",
            11: "RIGHTFIRE",
            12: "LEFTFIRE",
            13: "DOWNFIRE",
            14: "UPRIGHTFIRE",
            15: "UPLEFTFIRE",
            16: "DOWNRIGHTFIRE",
            17: "DOWNLEFTFIRE",
        }
        self.font = pygame.font.SysFont(self.cfg['Setting']['Font']['Family'], self.cfg['Setting']['Font']['Size'])
        self.font_height = self.font.get_height() * 1.2
        self.control_tick = 0
        self.fps = self.cfg['Setting']['FPS']

    def _setup_env(self):
        if not self.game:
            self.game = self.cfg['Game']
        self.env_name = "{}NoFrameskip-v3".format(self.game)
        self.env = gym.make(self.env_name)
        self.screen_height, self.screen_width, self.channel = self.env.reset().shape

    def _setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width * 2 + 400, self.screen_height * 2))
        pygame.display.set_caption("Arcade Learning Environment Human Interface")
        pygame.display.flip()
        self.clock = pygame.time.Clock()

    def _handle_event(self):
        end = False
        if self.control_tick == self.cfg['Key']['FrameSkip']:
            pressed = pygame.key.get_pressed()
            self.control_tick = 0
        else:
            pressed = [0 for _ in pygame.key.get_pressed()]
            self.control_tick += 1

        keys = 0
        if self.cfg['Key']['InvertUD']:
            keys |= pressed[getattr(pygame, self.cfg['Key']['UP'])] << 1
            keys |= pressed[getattr(pygame, self.cfg['Key']['DOWN'])]
        else:
            keys |= pressed[getattr(pygame, self.cfg['Key']['UP'])]
            keys |= pressed[getattr(pygame, self.cfg['Key']['DOWN'])] << 1
        if self.cfg['Key']['InvertLR']:
            keys |= pressed[getattr(pygame, self.cfg['Key']['LEFT'])] << 3
            keys |= pressed[getattr(pygame, self.cfg['Key']['RIGHT'])] << 2
        else:
            keys |= pressed[getattr(pygame, self.cfg['Key']['LEFT'])] << 2
            keys |= pressed[getattr(pygame, self.cfg['Key']['RIGHT'])] << 3
        keys |= pressed[getattr(pygame, self.cfg['Key']['FIRE'])] << 4
        action = self.key_map[keys]

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == getattr(pygame, self.cfg['Key']['QUIT']):
                    end = True
                if event.key == getattr(pygame, self.cfg['Key']['PAUSE']):
                    self.pause = not self.pause
            if event.type == pygame.QUIT:
                end = True
                break
        return end, action

    def _display_game(self, observation):
        observation = np.stack([observation[..., i].transpose() for i in range(self.channel)], axis=2)
        surf = pygame.pixelcopy.make_surface(observation)
        surf = pygame.transform.scale2x(surf)
        self.screen.blit(surf, (0, 0))

    def _display_info(self, action, reward, done=False):
        line_pos = 40
        text = self.font.render("Game: %s  FPS: %.1f" % (self.cfg['Game'], self.fps), 1, (132, 229, 105))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("Step: %d" % (self.frame_idx), 1, (132, 229, 105))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("Control: UP: %s" % (self.cfg['Key']['UP']), 1, (237, 217, 147))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("         DOWN: %s" % (self.cfg['Key']['DOWN']), 1, (237, 217, 147))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("         LEFT: %s" % (self.cfg['Key']['LEFT']), 1, (237, 217, 147))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("         RIGHT: %s" % (self.cfg['Key']['RIGHT']), 1, (237, 217, 147))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("         FIRE: %s" % (self.cfg['Key']['FIRE']), 1, (237, 217, 147))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("         PAUSE: %s" % (self.cfg['Key']['PAUSE']), 1, (237, 217, 147))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("         QUIT: %s" % (self.cfg['Key']['QUIT']), 1, (237, 217, 147))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("Info: ", 1, (71, 181, 224))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("      Current Action: " + self.action_meaning[action], 1, (208, 208, 255))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("      Total Reward: " + str(reward), 1, (208, 255, 255))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        if self.pause:
            text = self.font.render("Game Paused", 1, (224, 111, 71))
            self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
            line_pos += self.font_height

        if done:
            text = self.font.render("Game Over", 1, (224, 111, 71))
            self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
            line_pos += self.font_height

    def play(self, record=False):
        if record:
            replay = Memory(0, RGB_REPLAY_SHAPE)
        observation = self.env.reset()
        self.pause = False
        total_reward = 0
        self.frame_idx = 0
        while True:
            if self.pause:
                self.screen.fill((0, 0, 0))
                self._display_game(observation)
                end, action = self._handle_event()
                self._display_info(0, total_reward)
                pygame.display.flip()
                self.clock.tick(self.fps)
                if end:
                    return False
                continue
            self.screen.fill((0, 0, 0))
            self._display_game(observation)
            end, action = self._handle_event()
            if end:
                return False
            observation_, reward, done, info = self.env.step(action)
            if record:
                replay.append((observation, action, reward, done))
            total_reward += reward
            self.frame_idx += 1
            self._display_info(action, total_reward)
            pygame.display.flip()
            self.clock.tick(self.fps)
            observation = observation_
            if done:
                self.screen.fill((0, 0, 0))
                self._display_game(observation)
                self._display_info(action, total_reward, done)
                pygame.display.flip()
                if record:
                    replay_name = "%s_%d_%d.rep" % (self.game, total_reward, len(replay))
                    replay.save(get_path('tmp') + '/' + replay_name)
                for i in range(10):
                    self.clock.tick(self.fps)
                return True


class ReplayInterface(object):
    def __init__(self, game, cfg='RI.yml'):
        self.cfg = load_config(cfg)
        self.game = game
        self._setup_env()
        self._setup_pygame()
        self.action_meaning = {
            0: "NOOP",
            1: "FIRE",
            2: "UP",
            3: "RIGHT",
            4: "LEFT",
            5: "DOWN",
            6: "UPRIGHT",
            7: "UPLEFT",
            8: "DOWNRIGHT",
            9: "DOWNLEFT",
            10: "UPFIRE",
            11: "RIGHTFIRE",
            12: "LEFTFIRE",
            13: "DOWNFIRE",
            14: "UPRIGHTFIRE",
            15: "UPLEFTFIRE",
            16: "DOWNRIGHTFIRE",
            17: "DOWNLEFTFIRE",
        }
        self.font = pygame.font.SysFont(self.cfg['Setting']['Font']['Family'], self.cfg['Setting']['Font']['Size'])
        self.font_height = self.font.get_height() * 1.2
        self.fps = self.cfg['Setting']['FPS']

    def _setup_env(self):
        self.env_name = "{}NoFrameskip-v3".format(self.game)
        self.env = gym.make(self.env_name)
        self.screen_height, self.screen_width, self.channel = self.env.reset().shape

    def _setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width * 2 + 400, self.screen_height * 2))
        pygame.display.set_caption("Arcade Learning Environment Replay Interface")
        pygame.display.flip()
        self.clock = pygame.time.Clock()

    def _display_game(self, observation):
        observation = np.stack([observation[..., i].transpose() for i in range(self.channel)], axis=2)
        surf = pygame.pixelcopy.make_surface(observation)
        surf = pygame.transform.scale2x(surf)
        self.screen.blit(surf, (0, 0))

    def _display_info(self, action, reward, done=False):
        line_pos = 40
        text = self.font.render("Game: %s  FPS: %.1f" % (self.cfg['Game'], self.fps), 1, (132, 229, 105))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("Step: %d" % (self.frame_idx), 1, (132, 229, 105))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("Info: ", 1, (71, 181, 224))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("      Current Action: " + self.action_meaning[action], 1, (208, 208, 255))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        text = self.font.render("      Total Reward: " + str(reward), 1, (208, 255, 255))
        self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
        line_pos += self.font_height

        if self.pause:
            text = self.font.render("Game Paused", 1, (224, 111, 71))
            self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
            line_pos += self.font_height

        if done:
            text = self.font.render("Game Over", 1, (224, 111, 71))
            self.screen.blit(text, (self.screen_width * 2 + 50, line_pos))
            line_pos += self.font_height

    def _handle_event(self):
        end = False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == getattr(pygame, self.cfg['Key']['QUIT']):
                    end = True
                if event.key == getattr(pygame, self.cfg['Key']['PAUSE']):
                    self.pause = not self.pause
            if event.type == pygame.QUIT:
                end = True
                break
        return end

    def play(self, record):
        replay = Memory.load(record)
        replay.restore_block()
        self.pause = False
        total_reward = 0
        self.frame_idx = 0
        while self.frame_idx < len(replay):
            if self.pause:
                self.screen.fill((0, 0, 0))
                self._display_game(observation)
                end = self._handle_event()
                self._display_info(0, total_reward)
                pygame.display.flip()
                self.clock.tick(self.fps)
                if end:
                    return False
                continue
            observation, action, reward, done = replay[self.frame_idx]
            self.screen.fill((0, 0, 0))
            self._display_game(observation)
            end = self._handle_event()
            if end:
                return False
            total_reward += reward
            self._display_info(action, total_reward)
            pygame.display.flip()
            self.clock.tick(self.fps)
            self.frame_idx += 1
            if done:
                self.screen.fill((0, 0, 0))
                self._display_game(observation)
                self._display_info(action, total_reward, done)
                pygame.display.flip()
                for i in range(10):
                    self.clock.tick(self.fps)
                return True
