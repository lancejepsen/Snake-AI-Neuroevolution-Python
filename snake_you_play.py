import random
import time
import turtle

# =============================
# SOUND EFFECTS ONLY (Windows)
# =============================
try:
    import winsound

    def play_eat():
        winsound.Beep(950, 90)

    def play_die():
        winsound.Beep(300, 420)

except ImportError:
    def play_eat():
        pass

    def play_die():
        pass


# =============================
# GRID CONFIG
# =============================
CELL = 20
GRID_W = 45
GRID_H = 30

WIDTH = GRID_W * CELL
HEIGHT = GRID_H * CELL

OBSTACLE_COUNT = 18
FOOD_WALL_MARGIN = 2  # food won't spawn near walls

# Speed control
START_DELAY = 0.12
SPEED_MULT = 0.97
MIN_DELAY = 0.045

# =============================
# SCREEN
# =============================
screen = turtle.Screen()
screen.title("Snake — Arcade Grid (No Music)")
screen.bgcolor("black")
screen.setup(WIDTH, HEIGHT)
screen.tracer(0)

# =============================
# HELPERS
# =============================
def grid_to_xy(cell):
    cx, cy = cell
    return (
        (cx - GRID_W // 2) * CELL,
        (cy - GRID_H // 2) * CELL,
    )


def make_block(color):
    t = turtle.Turtle("square")
    t.color(color)
    t.penup()
    t.speed(0)
    return t


# =============================
# GAME STATE
# =============================
snake = []
obstacles = set()
food = None

direction = (1, 0)
game_over = False
pending_action = None

delay = START_DELAY
score = 0

# =============================
# DRAW OBJECTS
# =============================
snake_draw = []
obstacle_draw = []
food_draw = make_block("green")

score_hud = turtle.Turtle()
score_hud.hideturtle()
score_hud.color("cyan")
score_hud.penup()
score_hud.goto(-WIDTH // 2 + 12, HEIGHT // 2 - 32)

msg_hud = turtle.Turtle()
msg_hud.hideturtle()
msg_hud.color("cyan")
msg_hud.penup()
msg_hud.goto(0, HEIGHT // 2 - 32)


def update_score():
    score_hud.clear()
    score_hud.write(
        f"Score: {score}",
        align="left",
        font=("Consolas", 16, "normal"),
    )


def show_message(msg):
    msg_hud.clear()
    msg_hud.write(
        msg,
        align="center",
        font=("Consolas", 16, "normal"),
    )


# =============================
# SPAWNING
# =============================
def random_empty_cell(x_min, x_max, y_min, y_max):
    while True:
        c = (
            random.randrange(x_min, x_max + 1),
            random.randrange(y_min, y_max + 1),
        )
        if c not in snake and c not in obstacles:
            return c


def spawn_food():
    global food
    m = FOOD_WALL_MARGIN
    food = random_empty_cell(
        m, GRID_W - 1 - m,
        m, GRID_H - 1 - m,
    )
    food_draw.goto(*grid_to_xy(food))


def build_obstacles():
    obstacles.clear()
    for t in obstacle_draw:
        t.hideturtle()
    obstacle_draw.clear()

    cx = GRID_W // 2
    cy = GRID_H // 2

    protected = {
        (cx, cy),
        (cx - 1, cy),
        (cx - 2, cy),
        (cx + 1, cy),
        (cx + 2, cy),
    }

    while len(obstacles) < OBSTACLE_COUNT:
        c = random_empty_cell(0, GRID_W - 1, 0, GRID_H - 1)
        if c not in protected:
            obstacles.add(c)

    for c in obstacles:
        t = make_block("gray")
        t.goto(*grid_to_xy(c))
        obstacle_draw.append(t)


# =============================
# RESET / DEATH
# =============================
def reset_game():
    global snake, direction, game_over, pending_action, delay, score

    for t in snake_draw:
        t.hideturtle()
    snake_draw.clear()

    cx = GRID_W // 2
    cy = GRID_H // 2

    snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
    direction = (1, 0)
    game_over = False
    pending_action = None

    delay = START_DELAY
    score = 0

    for c in snake:
        t = make_block("white")
        t.goto(*grid_to_xy(c))
        snake_draw.append(t)

    build_obstacles()
    spawn_food()

    update_score()
    show_message("Arrow keys move | R replay | Q quit")


def die(reason):
    global game_over
    game_over = True
    play_die()
    show_message(f"💀 {reason} — R: Replay | Q: Quit")


# =============================
# CONTROLS
# =============================
def set_dir(dx, dy):
    global direction
    if not game_over and direction != (-dx, -dy):
        direction = (dx, dy)


def replay():
    global pending_action
    if game_over:
        pending_action = "replay"


def quit_game():
    global pending_action
    if game_over:
        pending_action = "quit"


screen.listen()
screen.onkeypress(lambda: set_dir(0, 1), "Up")
screen.onkeypress(lambda: set_dir(0, -1), "Down")
screen.onkeypress(lambda: set_dir(-1, 0), "Left")
screen.onkeypress(lambda: set_dir(1, 0), "Right")
screen.onkeypress(replay, "r")
screen.onkeypress(replay, "R")
screen.onkeypress(quit_game, "q")
screen.onkeypress(quit_game, "Q")


# =============================
# MAIN LOOP
# =============================
reset_game()

while True:
    screen.update()

    if pending_action == "replay":
        reset_game()
        time.sleep(0.15)
        continue

    if pending_action == "quit":
        break

    if game_over:
        time.sleep(0.05)
        continue

    hx, hy = snake[0]
    dx, dy = direction
    new_head = (hx + dx, hy + dy)

    # Wall collision
    if (
        new_head[0] < 0 or new_head[0] >= GRID_W
        or new_head[1] < 0 or new_head[1] >= GRID_H
    ):
        die("Hit wall")
        continue

    # Obstacle collision
    if new_head in obstacles:
        die("Hit obstacle")
        continue

    # Self collision
    if new_head in snake:
        die("Hit tail")
        continue

    snake.insert(0, new_head)

    # Eat food
    if new_head == food:
        score += 10
        update_score()
        play_eat()
        delay = max(MIN_DELAY, delay * SPEED_MULT)
        spawn_food()
    else:
        snake.pop()

    # Redraw snake
    for t in snake_draw:
        t.hideturtle()
    snake_draw.clear()

    for c in snake:
        t = make_block("white")
        t.goto(*grid_to_xy(c))
        snake_draw.append(t)

    time.sleep(delay)

turtle.bye()
