"""
Snake Neuroevolution by Lance Jepsen

You can play snake yourself by running snake_you_play.py and using
the arrow keys to control the snake.

Snake Neuroevolution is a genetic algorithm that learns to play
snake by evolving a neural network genome.

Controls:
- Space = pause
- Q = quit + save
- R = reset learning (deletes save file)
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import turtle

Cell = Tuple[int, int]

# ==========================================================
# SPEED / TRAINING MODE
# ==========================================================
HEADLESS_TRAINING = True
DEMO_EVERY_GENS = 15
DEMO_IF_BEST_AT_LEAST = 6
WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Common-random-numbers evaluation
SCENARIOS_PER_GEN = 2
SCENARIO_SEED_GAP = 7919

# Elite re-eval for stability
ELITE_REEVAL_EPISODES = 6
ELITE_REEVAL_SEED_GAP = 10007

# ==========================================================
# GRID (logical size) — renderer auto-scales
# ==========================================================
GRID_W, GRID_H = 34, 26

HUD_SPACE = 120
TOP_PADDING = 24
BOTTOM_PADDING = 80
SIDE_PADDING = 60

MIN_CELL = 14
BORDER_PEN_MIN = 2
BORDER_PEN_MAX = 6

# ==========================================================
# FEATURES + MEMORY
# ==========================================================
MEMORY_STEPS = 6

# Base features:
# danger(3)
# rays_to_block(3)
# tail_rays(self-only)(3)
# food_rays(visibility)(3)
# dir(4)
# food bool(4)
# food dx/dy (2)
BASE_FEATURES = 22
MEM_FEATURES = MEMORY_STEPS * 3
INPUT_SIZE = BASE_FEATURES + MEM_FEATURES  # 40

# ==========================================================
# OBSTACLES: static layout, ramp the number enabled
# ==========================================================
MAX_OBSTACLES = 14
START_OBSTACLES = 8
FOOD_MARGIN = 2

RAMP_TABLE = [
    (8, 14),
    (6, 12),
    (4, 10),
    (2, 9),
    (0, 8),
]

# ==========================================================
# MOVING-OBSTACLE CURRICULUM
# ==========================================================
STATIC_MOVE_EVERY = 10_000_000
MOVE_EVERY_EASY = 60
MOVE_EVERY_MED = 42
MOVE_EVERY_HARD = 28

START_MOVING_AT_SCORE = 8
INCREASE_MOVE_AT_SCORE = 11
STABILITY_GENS_REQUIRED = 6

# ==========================================================
# EVOLUTION
# ==========================================================
POP = 360
HIDDEN = 22

ELITISM = 18
TOURN_K = 9

MUT_RATE = 0.12
MUT_SIGMA = 0.45

EVAL_MAX_STEPS = 2600

# ==========================================================
# EXPLORATION (ε-greedy)
# ==========================================================
EPSILON_START = 0.18
EPSILON_END = 0.03
EPSILON_DECAY_GENS = 140


def epsilon_for_gen(gen: int) -> float:
    t = min(1.0, gen / float(EPSILON_DECAY_GENS))
    return EPSILON_START * (1.0 - t) + EPSILON_END * t


# ==========================================================
# TURTLE DEMO
# ==========================================================
DEMO_DELAY = 0.012


def demo_steps_for_score(best_score: int) -> int:
    """
    Increase demo length as the agent gets better.
    Keeps a hard cap to avoid infinite/very long demos.
    """
    if best_score < 3:
        return 400
    if best_score < 6:
        return 700
    if best_score < 10:
        return 1200
    if best_score < 15:
        return 2000
    if best_score < 25:
        return 3200
    return 5000


# ==========================================================
# SAVE / LOAD
# ==========================================================
SAVE_PATH = "snake_ai_progress.json"
AUTO_LOAD_IF_EXISTS = True
AUTO_SAVE_EVERY_GENS = 5
SAVE_VERSION = 14  # bumped due to demo logic + code versioning

# ==========================================================
# Renderer globals (computed at runtime)
# ==========================================================
CELL_SIZE = 22
GRID_Y_OFFSET = 0
SHAPE_SCALE = 1.0
BORDER_PEN_SIZE = 3

# ==========================================================
# WORKER GLOBALS (multiprocessing)
# ==========================================================
W_FULL_OBS: List[Cell] = []


# ==========================================================
# Snake Environment
# ==========================================================
@dataclass
class StepResult:
    alive: bool
    ate: bool
    score: int


class SnakeEnv:
    DIRS: List[Cell] = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R, U, L, D

    def __init__(
        self,
        *,
        rng: random.Random,
        full_obstacles: Sequence[Cell],
        active_obstacles: int,
        move_every: int,
        food_wall_margin: int,
    ) -> None:
        self.rng = rng
        self.full_obstacles = list(full_obstacles)
        self.active_obstacles = int(active_obstacles)
        self.move_every = int(move_every)
        self.food_wall_margin = int(food_wall_margin)

        self.max_steps_base = 260
        self.max_steps_per_food = 170

        self.snake: List[Cell] = []
        self.obstacles: Set[Cell] = set()
        self.food: Cell = (0, 0)
        self.dir_idx = 0
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.action_history: List[int] = [1] * MEMORY_STEPS
        self.epsilon: float = 0.0

    def reset(self) -> None:
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        cx = GRID_W // 2
        cy = GRID_H // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.dir_idx = 0
        self.action_history = [1] * MEMORY_STEPS

        self.obstacles = set(self.full_obstacles[: self.active_obstacles])
        self._spawn_food()

    def _spawn_food(self) -> None:
        m = self.food_wall_margin
        x_min, x_max = m, GRID_W - 1 - m
        y_min, y_max = m, GRID_H - 1 - m

        occupied = set(self.snake) | self.obstacles
        while True:
            c = (
                self.rng.randrange(x_min, x_max + 1),
                self.rng.randrange(y_min, y_max + 1),
            )
            if c not in occupied:
                self.food = c
                return

    def _is_blocked(self, cell: Cell) -> bool:
        x, y = cell
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return True
        if cell in self.obstacles:
            return True
        if cell in self.snake:
            return True
        return False

    def _try_move_one_obstacle(self) -> None:
        if not self.obstacles:
            return

        obs_list = list(self.obstacles)
        self.rng.shuffle(obs_list)

        occupied = set(self.snake) | {self.food}
        for o in obs_list:
            dirs = self.DIRS[:]
            self.rng.shuffle(dirs)
            for dx, dy in dirs:
                n = (o[0] + dx, o[1] + dy)
                if not (0 <= n[0] < GRID_W and 0 <= n[1] < GRID_H):
                    continue
                if n in self.obstacles or n in occupied:
                    continue
                self.obstacles.remove(o)
                self.obstacles.add(n)
                return

    def _distance_to_block(self, start: Cell, dx: int, dy: int) -> float:
        dist = 0
        x, y = start
        max_dist = max(GRID_W, GRID_H)

        while True:
            x += dx
            y += dy
            dist += 1
            if (
                x < 0
                or x >= GRID_W
                or y < 0
                or y >= GRID_H
                or (x, y) in self.obstacles
                or (x, y) in self.snake
            ):
                break

        return min(dist / max_dist, 1.0)

    def _distance_to_self_only(self, start: Cell, dx: int, dy: int) -> float:
        x, y = start
        max_dist = max(GRID_W, GRID_H)
        body = set(self.snake[1:])

        dist = 0
        while True:
            x += dx
            y += dy
            dist += 1

            if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
                return 1.0
            if (x, y) in body:
                return min(dist / max_dist, 1.0)

    def _food_ray_visibility(self, start: Cell, dx: int, dy: int) -> float:
        x, y = start
        max_dist = max(GRID_W, GRID_H)
        fx, fy = self.food
        dist = 0

        while True:
            x += dx
            y += dy
            dist += 1

            if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
                return 0.0

            if x == fx and y == fy:
                return max(0.0, 1.0 - (dist - 1) / float(max_dist))

    def get_features(self) -> np.ndarray:
        head = self.snake[0]
        hx, hy = head
        fx, fy = self.food

        left_idx = (self.dir_idx + 1) % 4
        right_idx = (self.dir_idx - 1) % 4

        dx, dy = self.DIRS[self.dir_idx]
        ldx, ldy = self.DIRS[left_idx]
        rdx, rdy = self.DIRS[right_idx]

        danger_straight = 1.0 if self._is_blocked((hx + dx, hy + dy)) else 0.0
        danger_left = 1.0 if self._is_blocked((hx + ldx, hy + ldy)) else 0.0
        danger_right = 1.0 if self._is_blocked((hx + rdx, hy + rdy)) else 0.0

        ray_straight = self._distance_to_block(head, dx, dy)
        ray_left = self._distance_to_block(head, ldx, ldy)
        ray_right = self._distance_to_block(head, rdx, rdy)

        tail_ray_straight = self._distance_to_self_only(head, dx, dy)
        tail_ray_left = self._distance_to_self_only(head, ldx, ldy)
        tail_ray_right = self._distance_to_self_only(head, rdx, rdy)

        food_ray_straight = self._food_ray_visibility(head, dx, dy)
        food_ray_left = self._food_ray_visibility(head, ldx, ldy)
        food_ray_right = self._food_ray_visibility(head, rdx, rdy)

        dir_left = 1.0 if self.dir_idx == 2 else 0.0
        dir_right = 1.0 if self.dir_idx == 0 else 0.0
        dir_up = 1.0 if self.dir_idx == 1 else 0.0
        dir_down = 1.0 if self.dir_idx == 3 else 0.0

        food_left = 1.0 if fx < hx else 0.0
        food_right = 1.0 if fx > hx else 0.0
        food_up = 1.0 if fy > hy else 0.0
        food_down = 1.0 if fy < hy else 0.0

        food_dx_norm = (fx - hx) / float(GRID_W)
        food_dy_norm = (fy - hy) / float(GRID_H)

        mem: List[float] = []
        for a in self.action_history:
            mem.extend([1.0 if a == 0 else 0.0, 1.0 if a == 1 else 0.0, 1.0 if a == 2 else 0.0])

        return np.array(
            [
                danger_straight,
                danger_left,
                danger_right,
                ray_straight,
                ray_left,
                ray_right,
                tail_ray_straight,
                tail_ray_left,
                tail_ray_right,
                food_ray_straight,
                food_ray_left,
                food_ray_right,
                dir_left,
                dir_right,
                dir_up,
                dir_down,
                food_left,
                food_right,
                food_up,
                food_down,
                food_dx_norm,
                food_dy_norm,
                *mem,
            ],
            dtype=np.float32,
        )

    def step(self, action: int) -> StepResult:
        self.steps += 1
        self.steps_since_food += 1

        self.action_history.append(action)
        if len(self.action_history) > MEMORY_STEPS:
            self.action_history.pop(0)

        if self.steps % self.move_every == 0:
            self._try_move_one_obstacle()

        if action == 0:
            self.dir_idx = (self.dir_idx + 1) % 4
        elif action == 2:
            self.dir_idx = (self.dir_idx - 1) % 4

        dx, dy = self.DIRS[self.dir_idx]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        if self._is_blocked(new_head):
            return StepResult(alive=False, ate=False, score=self.score)

        self.snake.insert(0, new_head)

        ate = new_head == self.food
        if ate:
            self.score += 1
            self.steps_since_food = 0
            self._spawn_food()
        else:
            self.snake.pop()

        max_steps = self.max_steps_base + self.score * self.max_steps_per_food
        if self.steps_since_food > max_steps:
            return StepResult(alive=False, ate=False, score=self.score)

        return StepResult(alive=True, ate=ate, score=self.score)


# ==========================================================
# Neural net genome (packed vector)
# ==========================================================
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def genome_num_params() -> int:
    return HIDDEN * INPUT_SIZE + HIDDEN + (3 * HIDDEN) + 3


def pack_genome(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.concatenate([w1.reshape(-1), b1.reshape(-1), w2.reshape(-1), b2.reshape(-1)]).astype(np.float32)


def unpack_genome(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vec = vec.astype(np.float32, copy=False)
    idx = 0

    w1_n = HIDDEN * INPUT_SIZE
    w1 = vec[idx: idx + w1_n].reshape((HIDDEN, INPUT_SIZE))
    idx += w1_n

    b1 = vec[idx: idx + HIDDEN]
    idx += HIDDEN

    w2_n = 3 * HIDDEN
    w2 = vec[idx: idx + w2_n].reshape((3, HIDDEN))
    idx += w2_n

    b2 = vec[idx: idx + 3]
    return w1, b1, w2, b2


def random_genome(rng: np.random.Generator) -> np.ndarray:
    w1 = rng.normal(0, 0.55, size=(HIDDEN, INPUT_SIZE)).astype(np.float32)
    b1 = rng.normal(0, 0.12, size=(HIDDEN,)).astype(np.float32)
    w2 = rng.normal(0, 0.55, size=(3, HIDDEN)).astype(np.float32)
    b2 = rng.normal(0, 0.05, size=(3,)).astype(np.float32) + rng.uniform(-0.03, 0.03, size=(3,)).astype(np.float32)
    return pack_genome(w1, b1, w2, b2)


def act_from_genome(genome_vec: np.ndarray, features: np.ndarray) -> int:
    w1, b1, w2, b2 = unpack_genome(genome_vec)
    h = relu(w1 @ features + b1)
    out = w2 @ h + b2
    return int(np.argmax(out))


def mutate(genome_vec: np.ndarray, rng: np.random.Generator, rate: float, sigma: float) -> np.ndarray:
    g = genome_vec.copy()
    mask = rng.random(g.shape) < rate
    g[mask] += rng.normal(0, sigma, size=g.shape)[mask].astype(np.float32)
    return g


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(a.shape) < 0.5
    child = a.copy()
    child[mask] = b[mask]
    return child


# ==========================================================
# Fitness (ANTI-CIRCLE v2)
# ==========================================================
def evaluate_once(genome_vec: np.ndarray, env: SnakeEnv) -> Tuple[float, int]:
    env.reset()
    fitness = 0.0

    recent: List[Cell] = []
    recent_set: Set[Cell] = set()
    recent_limit = 130

    last_dist: Optional[int] = None

    no_progress_steps = 0
    best_dist_so_far = 10**9

    last_action = 1
    oscillations = 0

    for _ in range(EVAL_MAX_STEPS):
        feats = env.get_features()

        action = act_from_genome(genome_vec, feats)
        if env.epsilon > 0.0 and env.rng.random() < env.epsilon:
            action = env.rng.randrange(3)

        hx, hy = env.snake[0]
        fx, fy = env.food
        if last_dist is None:
            last_dist = abs(hx - fx) + abs(hy - fy)

        res = env.step(action)
        fitness += 0.0005

        if res.ate:
            fitness += 260.0
            last_dist = None
            no_progress_steps = 0
            best_dist_so_far = 10**9
            oscillations = 0
        else:
            nhx, nhy = env.snake[0]
            new_dist = abs(nhx - fx) + abs(nhy - fy)

            if new_dist < last_dist:
                fitness += 0.55
            elif new_dist > last_dist:
                fitness -= 0.30
            last_dist = new_dist

            if new_dist < best_dist_so_far:
                best_dist_so_far = new_dist
                no_progress_steps = 0
            else:
                no_progress_steps += 1
                if no_progress_steps > 70:
                    fitness -= 0.06
                if no_progress_steps > 180:
                    fitness -= 3.0
                    break

        if action != 1:
            fitness -= 0.010

        if (action == 0 and last_action == 2) or (action == 2 and last_action == 0):
            oscillations += 1
            if oscillations > 10:
                fitness -= 0.04
        else:
            oscillations = max(0, oscillations - 1)
        last_action = action

        head = env.snake[0]
        if head in recent_set:
            fitness -= 0.40

        recent.append(head)
        recent_set.add(head)
        if len(recent) > recent_limit:
            old = recent.pop(0)
            if old not in recent:
                recent_set.discard(old)

        if not res.alive:
            fitness -= 44.0
            break

    fitness += env.score * 720.0
    return float(fitness), int(env.score)


# ==========================================================
# Multiprocessing worker
# ==========================================================
def _worker_init(full_obstacles_list: List[List[int]]) -> None:
    global W_FULL_OBS
    W_FULL_OBS = [tuple(x) for x in full_obstacles_list]


def _eval_worker(job: Tuple[bytes, int, int, int, float]) -> Tuple[float, int]:
    genome_bytes, move_every, seed, active_obstacles, eps = job
    genome_vec = np.frombuffer(genome_bytes, dtype=np.float32)

    env = SnakeEnv(
        rng=random.Random(int(seed)),
        full_obstacles=W_FULL_OBS,
        active_obstacles=int(active_obstacles),
        move_every=int(move_every),
        food_wall_margin=FOOD_MARGIN,
    )
    env.epsilon = float(eps)
    return evaluate_once(genome_vec, env)


# ==========================================================
# Evolution
# ==========================================================
def evolve_population(pop: List[np.ndarray], fitness: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    idx = np.argsort(fitness)[::-1]
    elites = [pop[int(i)].copy() for i in idx[:ELITISM]]

    def pick() -> np.ndarray:
        cand = rng.choice(idx, size=TOURN_K, replace=False)
        best_i = int(cand[np.argmax(fitness[cand])])
        return pop[best_i]

    new_pop: List[np.ndarray] = elites[:]
    while len(new_pop) < len(pop):
        p1 = pick()
        p2 = pick()
        child = crossover(p1, p2, rng)
        child = mutate(child, rng, rate=MUT_RATE, sigma=MUT_SIGMA)
        new_pop.append(child)

    return new_pop


# ==========================================================
# Save / Load utilities
# ==========================================================
def _b64_encode_vec(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")


def _b64_decode_vec(s: str) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).copy()


def save_state(
    path: str,
    *,
    generation: int,
    best_ever: int,
    current_move_every: int,
    rolling_best_scores: List[int],
    base_seed: int,
    full_obstacles: List[Cell],
    population: List[np.ndarray],
) -> None:
    payload = {
        "version": SAVE_VERSION,
        "generation": generation,
        "best_ever": best_ever,
        "current_move_every": current_move_every,
        "rolling_best_scores": rolling_best_scores,
        "base_seed": base_seed,
        "grid_w": GRID_W,
        "grid_h": GRID_H,
        "max_obstacles": MAX_OBSTACLES,
        "start_obstacles": START_OBSTACLES,
        "food_margin": FOOD_MARGIN,
        "memory_steps": MEMORY_STEPS,
        "input_size": INPUT_SIZE,
        "hidden": HIDDEN,
        "pop_size": len(population),
        "genome_params": genome_num_params(),
        "full_obstacles": [list(x) for x in full_obstacles],
        "population_b64": [_b64_encode_vec(g) for g in population],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def load_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_progress_file(path: str) -> bool:
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ==========================================================
# Layout / rendering
# ==========================================================
def compute_layout(screen: turtle.Screen) -> None:
    global CELL_SIZE, GRID_Y_OFFSET, SHAPE_SCALE, BORDER_PEN_SIZE

    win_w = int(screen.window_width())
    win_h = int(screen.window_height())

    usable_w = max(200, win_w - 2 * SIDE_PADDING)
    usable_h = max(200, win_h - HUD_SPACE - TOP_PADDING - BOTTOM_PADDING)

    cell_w = usable_w // GRID_W
    cell_h = usable_h // GRID_H

    CELL_SIZE = max(MIN_CELL, int(min(cell_w, cell_h)))
    SHAPE_SCALE = CELL_SIZE / 20.0
    BORDER_PEN_SIZE = int(max(BORDER_PEN_MIN, min(BORDER_PEN_MAX, CELL_SIZE * 0.12)))

    top_limit = (win_h / 2) - HUD_SPACE - TOP_PADDING
    bottom_limit = (-win_h / 2) + BOTTOM_PADDING
    GRID_Y_OFFSET = int((top_limit + bottom_limit) / 2.0)

    print(
        f"[Layout] window={win_w}x{win_h} cell={CELL_SIZE} "
        f"grid_px={GRID_W*CELL_SIZE}x{GRID_H*CELL_SIZE} offset_y={GRID_Y_OFFSET}"
    )


def grid_to_xy(cell: Cell) -> Tuple[int, int]:
    cx, cy = cell
    x = (cx - GRID_W // 2) * CELL_SIZE
    y = (cy - GRID_H // 2) * CELL_SIZE + GRID_Y_OFFSET
    return (x, y)


def make_turtle_block(color: str, shape: str = "square") -> turtle.Turtle:
    t = turtle.Turtle(shape)
    t.color(color)
    t.penup()
    t.speed(0)
    if shape in ("square", "circle"):
        t.shapesize(stretch_wid=SHAPE_SCALE, stretch_len=SHAPE_SCALE)
    return t


def draw_border() -> turtle.Turtle:
    border = turtle.Turtle()
    border.hideturtle()
    border.speed(0)
    border.color("white")
    border.pensize(BORDER_PEN_SIZE)
    border.penup()

    half_w = (GRID_W * CELL_SIZE) / 2.0
    half_h = (GRID_H * CELL_SIZE) / 2.0

    left = -half_w - (CELL_SIZE / 2.0)
    right = half_w - (CELL_SIZE / 2.0)
    bottom = -half_h - (CELL_SIZE / 2.0) + GRID_Y_OFFSET
    top = half_h - (CELL_SIZE / 2.0) + GRID_Y_OFFSET

    border.goto(left, bottom)
    border.pendown()
    border.goto(right, bottom)
    border.goto(right, top)
    border.goto(left, top)
    border.goto(left, bottom)
    border.penup()
    return border


# ==========================================================
# Deterministic obstacle layout
# ==========================================================
def generate_full_obstacles(seed: int) -> List[Cell]:
    r = random.Random(seed)
    cx = GRID_W // 2
    cy = GRID_H // 2
    protected = {(cx, cy), (cx - 1, cy), (cx - 2, cy)}

    full: List[Cell] = []
    used: Set[Cell] = set(protected)

    while len(full) < MAX_OBSTACLES:
        c = (r.randrange(GRID_W), r.randrange(GRID_H))
        if c in used:
            continue
        used.add(c)
        full.append(c)

    return full


def ramp_obstacles_from_best(best_score: int) -> int:
    for threshold, count in RAMP_TABLE:
        if best_score >= threshold:
            return count
    return START_OBSTACLES


def sustained_at_or_above(scores: List[int], threshold: int) -> bool:
    if len(scores) < STABILITY_GENS_REQUIRED:
        return False
    tail = scores[-STABILITY_GENS_REQUIRED:]
    return all(s >= threshold for s in tail)


def phase_text(move_every: int) -> str:
    if move_every == STATIC_MOVE_EVERY:
        return "STATIC"
    return f"MOVING(every {move_every})"


# ==========================================================
# MAIN
# ==========================================================
def main() -> None:
    reset_requested = "--reset" in sys.argv
    init_rng = np.random.default_rng(7)
    evo_rng = np.random.default_rng(777)

    generation = 0
    best_ever = 0
    current_move_every = STATIC_MOVE_EVERY
    rolling_best_scores: List[int] = []
    base_seed = 123

    full_obstacles: List[Cell]
    population: List[np.ndarray]

    def init_fresh_state() -> Tuple[int, int, int, List[int], int, List[Cell], List[np.ndarray]]:
        fresh_generation = 0
        fresh_best_ever = 0
        fresh_move_every = STATIC_MOVE_EVERY
        fresh_rolling: List[int] = []
        fresh_base_seed = 123
        full_obs = generate_full_obstacles(fresh_base_seed)
        fresh_population = [random_genome(init_rng) for _ in range(POP)]
        return (
            fresh_generation,
            fresh_best_ever,
            fresh_move_every,
            fresh_rolling,
            fresh_base_seed,
            full_obs,
            fresh_population,
        )

    # Load / init
    if reset_requested:
        generation, best_ever, current_move_every, rolling_best_scores, base_seed, full_obstacles, population = init_fresh_state()
    elif os.path.exists(SAVE_PATH):
        try:
            state = load_state(SAVE_PATH)

            if int(state.get("version", -1)) != SAVE_VERSION:
                raise ValueError(f"Save incompatible: version mismatch (need {SAVE_VERSION}).")
            if int(state.get("input_size", -1)) != INPUT_SIZE:
                raise ValueError("Save incompatible: input_size mismatch.")
            if int(state.get("hidden", -1)) != HIDDEN:
                raise ValueError("Save incompatible: hidden mismatch.")
            if int(state.get("genome_params", -1)) != genome_num_params():
                raise ValueError("Save incompatible: genome_params mismatch.")

            generation = int(state.get("generation", 0))
            best_ever = int(state.get("best_ever", 0))
            current_move_every = int(state.get("current_move_every", STATIC_MOVE_EVERY))
            rolling_best_scores = list(state.get("rolling_best_scores", []))
            base_seed = int(state.get("base_seed", 123))

            full_obstacles = [tuple(x) for x in state.get("full_obstacles", [])]
            if len(full_obstacles) != MAX_OBSTACLES:
                full_obstacles = generate_full_obstacles(base_seed)

            pop_b64 = list(state.get("population_b64", []))
            population = [_b64_decode_vec(s) for s in pop_b64]
            if len(population) != POP:
                population = [random_genome(init_rng) for _ in range(POP)]

            print(f"[Loaded] {SAVE_PATH} (gen={generation}, best_ever={best_ever})")
        except Exception as e:
            print(f"[Load failed] {e} — starting fresh.")
            generation, best_ever, current_move_every, rolling_best_scores, base_seed, full_obstacles, population = init_fresh_state()
    else:
        generation, best_ever, current_move_every, rolling_best_scores, base_seed, full_obstacles, population = init_fresh_state()

    # Turtle window
    screen = turtle.Screen()
    screen.title("Snake AI by Lance Jepsen | Space=pause | Q=quit+save | R=reset")
    screen.bgcolor("black")
    screen.setup(width=1.0, height=1.0)
    screen.tracer(0)

    compute_layout(screen)
    _border = draw_border()

    win_h = int(screen.window_height())
    hud_y = (win_h / 2) - (TOP_PADDING + 34)
    info_y = hud_y - 28
    msg_y = info_y - 26

    hud = turtle.Turtle()
    hud.hideturtle()
    hud.color("cyan")
    hud.penup()
    hud.goto(0, hud_y)

    info = turtle.Turtle()
    info.hideturtle()
    info.color("cyan")
    info.penup()
    info.goto(0, info_y)

    msg = turtle.Turtle()
    msg.hideturtle()
    msg.color("yellow")
    msg.penup()
    msg.goto(0, msg_y)

    seg_turtles: Dict[int, turtle.Turtle] = {}
    obstacle_turtles: List[turtle.Turtle] = []
    food_t = make_turtle_block("green", "circle")

    paused = False
    should_quit = False
    reset_pending = False

    def toggle_pause() -> None:
        nonlocal paused
        paused = not paused

    def request_quit() -> None:
        nonlocal should_quit
        should_quit = True

    def request_reset_learning() -> None:
        nonlocal reset_pending
        reset_pending = True

    screen.listen()
    screen.onkeypress(toggle_pause, "space")
    screen.onkeypress(request_quit, "q")
    screen.onkeypress(request_quit, "Q")
    screen.onkeypress(request_reset_learning, "r")
    screen.onkeypress(request_reset_learning, "R")

    def draw_obstacles(active_count: int, env: Optional[SnakeEnv] = None) -> None:
        nonlocal obstacle_turtles
        for t in obstacle_turtles:
            t.hideturtle()
        obstacle_turtles = []

        cells = full_obstacles[:active_count] if env is None else list(env.obstacles)
        for c in cells:
            t = make_turtle_block("gray", "square")
            t.goto(*grid_to_xy(c))
            obstacle_turtles.append(t)

    def sync_snake(env: SnakeEnv) -> None:
        for i, cell in enumerate(env.snake):
            if i not in seg_turtles:
                seg_turtles[i] = make_turtle_block("white", "square")
            seg_turtles[i].goto(*grid_to_xy(cell))
        for i in list(seg_turtles.keys()):
            if i >= len(env.snake):
                seg_turtles[i].hideturtle()
                del seg_turtles[i]

    def clear_drawings() -> None:
        for t in list(seg_turtles.values()):
            t.hideturtle()
        seg_turtles.clear()
        for t in obstacle_turtles:
            t.hideturtle()
        obstacle_turtles.clear()
        food_t.hideturtle()

    # Multiprocessing pool
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=WORKERS,
        initializer=_worker_init,
        initargs=([list(x) for x in full_obstacles],),
    )

    try:
        last_demo_best = -1

        while True:
            if should_quit:
                try:
                    save_state(
                        SAVE_PATH,
                        generation=generation,
                        best_ever=best_ever,
                        current_move_every=current_move_every,
                        rolling_best_scores=rolling_best_scores,
                        base_seed=base_seed,
                        full_obstacles=full_obstacles,
                        population=population,
                    )
                    print(f"[Saved] {SAVE_PATH}")
                except Exception as e:
                    print(f"[Save failed] {e}")
                turtle.bye()
                return

            if reset_pending:
                deleted = delete_progress_file(SAVE_PATH)
                generation, best_ever, current_move_every, rolling_best_scores, base_seed, full_obstacles, population = init_fresh_state()

                reset_pending = False
                paused = False

                pool.close()
                pool.join()
                pool = ctx.Pool(
                    processes=WORKERS,
                    initializer=_worker_init,
                    initargs=([list(x) for x in full_obstacles],),
                )

                msg.clear()
                msg.write(
                    f"RESET: learning restarted ({'deleted save file' if deleted else 'no save file found'})",
                    align="center",
                    font=("Consolas", 12, "normal"),
                )
                screen.update()
                time.sleep(0.7)
                msg.clear()
                clear_drawings()

            if paused:
                info.clear()
                info.write(
                    "PAUSED (space resumes) | Q quits+save | R resets learning",
                    align="center",
                    font=("Consolas", 14, "normal"),
                )
                screen.update()
                time.sleep(0.08)
                continue

            generation += 1
            eps = epsilon_for_gen(generation)

            last_best = rolling_best_scores[-1] if rolling_best_scores else 0
            active_obstacles = ramp_obstacles_from_best(last_best)

            if current_move_every == STATIC_MOVE_EVERY and sustained_at_or_above(rolling_best_scores, START_MOVING_AT_SCORE):
                current_move_every = MOVE_EVERY_EASY
            elif current_move_every == MOVE_EVERY_EASY and sustained_at_or_above(rolling_best_scores, INCREASE_MOVE_AT_SCORE):
                current_move_every = MOVE_EVERY_MED
            elif current_move_every == MOVE_EVERY_MED and sustained_at_or_above(rolling_best_scores, INCREASE_MOVE_AT_SCORE + 2):
                current_move_every = MOVE_EVERY_HARD

            scenario_seeds = [
                base_seed + generation * 100000 + k * SCENARIO_SEED_GAP
                for k in range(SCENARIOS_PER_GEN)
            ]

            jobs: List[Tuple[bytes, int, int, int, float]] = []
            job_map: List[int] = []

            for i, g in enumerate(population):
                gb = g.tobytes()
                for s in scenario_seeds:
                    jobs.append((gb, int(current_move_every), int(s), int(active_obstacles), float(eps)))
                    job_map.append(i)

            results = pool.map(_eval_worker, jobs)

            fitness_acc = np.zeros(POP, dtype=np.float32)
            score_best = np.zeros(POP, dtype=np.int32)

            for gi, (fit, score) in zip(job_map, results):
                fitness_acc[gi] += float(fit)
                score_best[gi] = max(score_best[gi], int(score))

            fitness_arr = fitness_acc / float(SCENARIOS_PER_GEN)
            scores_arr = score_best

            idx_sorted = np.argsort(fitness_arr)[::-1]
            top_idx = idx_sorted[:ELITISM]

            elite_seeds = [
                base_seed + 9000000 + generation * 10000 + k * ELITE_REEVAL_SEED_GAP
                for k in range(ELITE_REEVAL_EPISODES)
            ]

            for i in top_idx:
                gi = int(i)
                g = population[gi]
                fits = []
                scs = []
                for s in elite_seeds:
                    env = SnakeEnv(
                        rng=random.Random(int(s)),
                        full_obstacles=full_obstacles,
                        active_obstacles=active_obstacles,
                        move_every=current_move_every,
                        food_wall_margin=FOOD_MARGIN,
                    )
                    env.epsilon = float(eps)
                    f, sc = evaluate_once(g, env)
                    fits.append(f)
                    scs.append(sc)
                fitness_arr[gi] = float(np.mean(fits))
                scores_arr[gi] = int(max(scs))

            best_i = int(np.argmax(fitness_arr))
            gen_best_score = int(scores_arr[best_i])
            avg_score = float(np.mean(scores_arr))
            best_ever = max(best_ever, gen_best_score)

            best_genome_for_demo = population[best_i].copy()

            rolling_best_scores.append(gen_best_score)
            if len(rolling_best_scores) > 14:
                rolling_best_scores.pop(0)

            phase = f"{phase_text(current_move_every)} | Obst={active_obstacles}/{MAX_OBSTACLES} | eps={eps:.3f}"
            print(f"Gen {generation:03d} | best={gen_best_score:2d} | avg={avg_score:.2f} | {phase} | best_ever={best_ever}")

            population = evolve_population(population, fitness_arr, evo_rng)

            if generation % AUTO_SAVE_EVERY_GENS == 0:
                try:
                    save_state(
                        SAVE_PATH,
                        generation=generation,
                        best_ever=best_ever,
                        current_move_every=current_move_every,
                        rolling_best_scores=rolling_best_scores,
                        base_seed=base_seed,
                        full_obstacles=full_obstacles,
                        population=population,
                    )
                    print(f"[Auto-saved] {SAVE_PATH}")
                except Exception as e:
                    print(f"[Auto-save failed] {e}")

            hud.clear()
            hud.write(
                f"Gen {generation} | Best {gen_best_score} | Avg {avg_score:.2f} | BestEver {best_ever}",
                align="center",
                font=("Consolas", 14, "normal"),
            )
            info.clear()
            info.write(
                f"{'HEADLESS' if HEADLESS_TRAINING else 'LIVE'} | {phase} | Space=pause | Q=quit+save | R=reset",
                align="center",
                font=("Consolas", 12, "normal"),
            )
            screen.update()

            show_demo = (
                (generation % DEMO_EVERY_GENS == 0)
                or (gen_best_score >= DEMO_IF_BEST_AT_LEAST)
                or (gen_best_score > last_demo_best)
            )
            if HEADLESS_TRAINING and not show_demo:
                continue

            last_demo_best = max(last_demo_best, gen_best_score)

            max_demo_steps = demo_steps_for_score(gen_best_score)

            clear_drawings()

            env_vis = SnakeEnv(
                rng=random.Random(int(scenario_seeds[0])),
                full_obstacles=full_obstacles,
                active_obstacles=active_obstacles,
                move_every=current_move_every,
                food_wall_margin=FOOD_MARGIN,
            )
            env_vis.epsilon = 0.0
            env_vis.reset()

            draw_obstacles(active_obstacles, env_vis if current_move_every != STATIC_MOVE_EVERY else None)

            food_t.showturtle()
            food_t.goto(*grid_to_xy(env_vis.food))

            steps = 0
            while steps < max_demo_steps:
                if should_quit or reset_pending:
                    break

                feats = env_vis.get_features()
                action = act_from_genome(best_genome_for_demo, feats)
                res = env_vis.step(action)

                sync_snake(env_vis)
                food_t.goto(*grid_to_xy(env_vis.food))

                if current_move_every != STATIC_MOVE_EVERY and (env_vis.steps % max(6, current_move_every) == 0):
                    draw_obstacles(active_obstacles, env_vis)

                # Show demo progress
                info.clear()
                info.write(
                    f"DEMO | steps {steps}/{max_demo_steps} | score {env_vis.score} | {phase_text(current_move_every)}",
                    align="center",
                    font=("Consolas", 12, "normal"),
                )

                screen.update()

                if not res.alive:
                    msg.clear()
                    msg.write(
                        f"💀 Demo died at score {env_vis.score} — next generation...",
                        align="center",
                        font=("Consolas", 13, "normal"),
                    )
                    screen.update()
                    time.sleep(0.30)
                    msg.clear()
                    break

                steps += 1
                time.sleep(DEMO_DELAY)

    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
