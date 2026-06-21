"""
Q-Learning ile Deneme-Yanılma Öğrenen Bot Simülasyonu
======================================================
Pygame ile görsel bir 2D ızgara ortamında, bir bot rastgele hareketlerle
başlayıp Q-learning algoritması ile hedefe en kısa yoldan ulaşmayı öğrenir.

Kontroller:
    SPACE       : Simülasyonu duraklat / devam ettir
    YUKARI OK   : Simülasyonu hızlandır (FPS artır)
    AŞAGI OK    : Simülasyonu yavaşlat (FPS azalt)
    F           : Süper hızlı mod (her karede çok adım) aç/kapat
    R           : Q-tablosunu sıfırla, baştan öğren
    Q veya ESC  : Çıkış

Gereksinim: pip install pygame numpy
"""

import sys
import random
import numpy as np
import pygame

# ---------------- AYARLAR ----------------
WINDOW_SIZE = 600         # pencere kenar uzunluğu (px)
GRID_SIZE = 20            # ızgara: GRID_SIZE x GRID_SIZE
CELL = WINDOW_SIZE // GRID_SIZE
INFO_HEIGHT = 120         # alttaki bilgi paneli yüksekliği

# Q-learning hiperparametreleri
ALPHA = 0.1               # öğrenme oranı
GAMMA = 0.9               # gelecek ödül indirim faktörü
EPSILON_START = 1.0       # başlangıçta tamamen rastgele
EPSILON_MIN = 0.05        # minimum keşif oranı
EPSILON_DECAY = 0.995     # her bölümde epsilon * decay

# Ödüller
REWARD_GOAL = 100.0
REWARD_OBSTACLE = -50.0
REWARD_STEP = -1.0        # her adım küçük ceza -> kısa yolu özendirir
REWARD_CLOSER = 0.5       # hedefe yaklaşınca ufak bonus
REWARD_FARTHER = -0.5     # hedeften uzaklaşınca ufak ceza

MAX_STEPS_PER_EPISODE = 200

# Aksiyonlar: 0=yukarı, 1=aşağı, 2=sol, 3=sağ
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_NAMES = ["YUKARI", "ASAGI", "SOL", "SAG"]

# Renkler
BG = (28, 30, 38)
GRID_COLOR = (50, 54, 65)
BOT_COLOR = (60, 140, 240)
GOAL_COLOR = (80, 200, 100)
OBSTACLE_COLOR = (220, 70, 70)
TEXT_COLOR = (235, 235, 235)
SUBTEXT_COLOR = (170, 175, 185)
PATH_COLOR = (90, 90, 120)
POLICY_COLOR = (255, 215, 0)


# ---------------- ORTAM ----------------
class GridEnv:
    """Q-learning için basit ızgara ortamı."""

    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        self.obstacles = self._build_obstacles()
        self.reset()

    def _build_obstacles(self):
        """Bir kaç duvar/engel kur. Start ve goal serbest kalmalı."""
        obs = set()
        # Birkaç dikey/yatay duvar parçası
        for y in range(4, 12):
            obs.add((6, y))
        for x in range(8, 16):
            obs.add((x, 8))
        for y in range(12, 18):
            obs.add((13, y))
        for x in range(2, 8):
            obs.add((x, 15))

        # Start ve goal'u temizle
        obs.discard(self.start)
        obs.discard(self.goal)
        return obs

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.pos

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action):
        """Bir aksiyon uygula -> (yeni_durum, odul, bitti_mi, basarili_mi)."""
        dx, dy = ACTIONS[action]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy

        # Sınır kontrolü: dışarı çıkarsa cezalandır, yerinde kal
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            self.steps += 1
            done = self.steps >= MAX_STEPS_PER_EPISODE
            return self.pos, REWARD_OBSTACLE / 2, done, False

        # Engel kontrolü
        if (nx, ny) in self.obstacles:
            self.steps += 1
            done = self.steps >= MAX_STEPS_PER_EPISODE
            # Engele girmesin, yerinde kalsın
            return self.pos, REWARD_OBSTACLE, done, False

        # Geçerli hareket
        old_dist = self._manhattan(self.pos, self.goal)
        self.pos = (nx, ny)
        new_dist = self._manhattan(self.pos, self.goal)
        self.steps += 1

        # Hedefe ulaştı mı?
        if self.pos == self.goal:
            return self.pos, REWARD_GOAL, True, True

        # Adım ödülü + yakınlaşma/uzaklaşma şekillendirmesi
        reward = REWARD_STEP
        if new_dist < old_dist:
            reward += REWARD_CLOSER
        elif new_dist > old_dist:
            reward += REWARD_FARTHER

        done = self.steps >= MAX_STEPS_PER_EPISODE
        return self.pos, reward, done, False


# ---------------- Q-LEARNING AJANI ----------------
class QLearningAgent:
    def __init__(self, grid_size, n_actions=4):
        self.grid_size = grid_size
        self.n_actions = n_actions
        # Q-tablosu: [x][y][action]
        self.Q = np.zeros((grid_size, grid_size, n_actions), dtype=np.float32)
        self.epsilon = EPSILON_START

    def reset_q(self):
        self.Q.fill(0.0)
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        """Epsilon-greedy politika."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        x, y = state
        q_vals = self.Q[x, y]
        # En yüksek Q'ya sahip birden fazla aksiyon varsa rastgele seç
        max_q = np.max(q_vals)
        best = [a for a, v in enumerate(q_vals) if v == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, done):
        x, y = state
        nx, ny = next_state
        current_q = self.Q[x, y, action]
        target = reward if done else reward + GAMMA * np.max(self.Q[nx, ny])
        self.Q[x, y, action] = current_q + ALPHA * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def best_action(self, state):
        x, y = state
        return int(np.argmax(self.Q[x, y]))


# ---------------- ÇİZİM ----------------
def draw_grid(surface):
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(surface, GRID_COLOR, (0, i * CELL),
                         (WINDOW_SIZE, i * CELL), 1)
        pygame.draw.line(surface, GRID_COLOR, (i * CELL, 0),
                         (i * CELL, WINDOW_SIZE), 1)


def draw_obstacles(surface, env):
    for (x, y) in env.obstacles:
        rect = pygame.Rect(x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2)
        pygame.draw.rect(surface, OBSTACLE_COLOR, rect, border_radius=4)


def draw_goal(surface, env):
    x, y = env.goal
    rect = pygame.Rect(x * CELL + 4, y * CELL + 4, CELL - 8, CELL - 8)
    pygame.draw.rect(surface, GOAL_COLOR, rect, border_radius=6)


def draw_bot(surface, env):
    x, y = env.pos
    cx = x * CELL + CELL // 2
    cy = y * CELL + CELL // 2
    pygame.draw.circle(surface, BOT_COLOR, (cx, cy), CELL // 2 - 3)


def draw_policy_arrows(surface, agent, env):
    """Şu anki Q tablosundan öğrenilen politikayı oklarla göster."""
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) in env.obstacles or (x, y) == env.goal:
                continue
            q_vals = agent.Q[x, y]
            if np.all(q_vals == 0):
                continue
            a = int(np.argmax(q_vals))
            dx, dy = ACTIONS[a]
            cx = x * CELL + CELL // 2
            cy = y * CELL + CELL // 2
            end = (cx + dx * (CELL // 3), cy + dy * (CELL // 3))
            pygame.draw.line(surface, POLICY_COLOR, (cx, cy), end, 2)
            pygame.draw.circle(surface, POLICY_COLOR, end, 2)


def draw_info(surface, font, small_font, stats):
    """Alttaki bilgi paneli."""
    panel = pygame.Rect(0, WINDOW_SIZE, WINDOW_SIZE, INFO_HEIGHT)
    pygame.draw.rect(surface, (20, 22, 28), panel)
    pygame.draw.line(surface, GRID_COLOR, (0, WINDOW_SIZE),
                     (WINDOW_SIZE, WINDOW_SIZE), 2)

    lines = [
        f"Deneme (Episode): {stats['episode']}    "
        f"Basari: {stats['success']}    "
        f"Basari Orani: {stats['success_rate']:.1f}%",
        f"Son Bolum Adim: {stats['last_steps']}    "
        f"En Az Adim: {stats['best_steps']}    "
        f"Toplam Odul (son): {stats['last_reward']:.1f}",
        f"Epsilon (kesif): {stats['epsilon']:.3f}    "
        f"FPS: {stats['fps']}    "
        f"{'DURAKLATILDI' if stats['paused'] else ('TURBO MOD' if stats['turbo'] else 'CALISIYOR')}",
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, TEXT_COLOR)
        surface.blit(txt, (12, WINDOW_SIZE + 8 + i * 22))

    help_text = "SPACE: duraklat | YUKARI/ASAGI: hiz | F: turbo | R: sifirla | ESC: cikis"
    htxt = small_font.render(help_text, True, SUBTEXT_COLOR)
    surface.blit(htxt, (12, WINDOW_SIZE + INFO_HEIGHT - 22))


# ---------------- ANA DÖNGÜ ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + INFO_HEIGHT))
    pygame.display.set_caption("Q-Learning Bot - Deneme Yanilma ile Ogrenen Bot")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 13)

    env = GridEnv(GRID_SIZE)
    agent = QLearningAgent(GRID_SIZE)

    # İstatistikler
    episode = 0
    success_count = 0
    last_steps = 0
    best_steps = None
    last_reward = 0.0
    episode_reward = 0.0
    paused = False
    turbo = False
    fps = 30
    state = env.reset()

    running = True
    while running:
        # ---------- OLAYLAR ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    fps = min(240, fps + 10)
                elif event.key == pygame.K_DOWN:
                    fps = max(2, fps - 10)
                elif event.key == pygame.K_f:
                    turbo = not turbo
                elif event.key == pygame.K_r:
                    agent.reset_q()
                    env = GridEnv(GRID_SIZE)
                    state = env.reset()
                    episode = 0
                    success_count = 0
                    last_steps = 0
                    best_steps = None
                    last_reward = 0.0
                    episode_reward = 0.0

        # ---------- SİMÜLASYON ADIMI ----------
        if not paused:
            # Turbo modda her ekran karesinde çok adım at -> hızlı öğrenme
            steps_this_frame = 500 if turbo else 1
            for _ in range(steps_this_frame):
                action = agent.choose_action(state)
                next_state, reward, done, success = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if done:
                    episode += 1
                    last_steps = env.steps
                    last_reward = episode_reward
                    if success:
                        success_count += 1
                        if best_steps is None or last_steps < best_steps:
                            best_steps = last_steps
                    agent.decay_epsilon()
                    state = env.reset()
                    episode_reward = 0.0

        # ---------- ÇİZİM ----------
        screen.fill(BG)
        # Oyun alanı
        play_area = pygame.Rect(0, 0, WINDOW_SIZE, WINDOW_SIZE)
        pygame.draw.rect(screen, BG, play_area)

        draw_grid(screen)
        draw_policy_arrows(screen, agent, env)
        draw_obstacles(screen, env)
        draw_goal(screen, env)
        draw_bot(screen, env)

        success_rate = (success_count / episode * 100.0) if episode > 0 else 0.0
        stats = {
            "episode": episode,
            "success": success_count,
            "success_rate": success_rate,
            "last_steps": last_steps,
            "best_steps": best_steps if best_steps is not None else "-",
            "last_reward": last_reward,
            "epsilon": agent.epsilon,
            "fps": fps if not turbo else f"{fps} (TURBO)",
            "paused": paused,
            "turbo": turbo,
        }
        draw_info(screen, font, small_font, stats)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
