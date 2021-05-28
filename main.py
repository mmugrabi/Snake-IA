#Made with lots of clumsiness and love by Jean Pierre HUYNH.
import pygame, random, time
import torch.optim as optim

TITLE = "snAIke!"
FPS = 30
BLACK = (50, 50, 50)
GREY = (120, 120, 120)
WHITE = (200, 200, 200)
YELLOW = (200, 200, 50)
RED = (200, 50, 50)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
UP = (-1, 0)
SNAKE_CHAR = '+'
EMPTY_CHAR = ' '
WALL_CHAR = '#'
FOOD_CHAR = '@'

class SnakeGame:
    def __init__(self):
        self.run = True
        self.rows = 20
        self.columns = 20
        self.grid = [[' ' for j in range(100)] for i in range(100)]
        self.snake = []
        self.previous_move = None
        self.next_move = None
        self.food = None
        self.alive = False
        self.score = 0
        self.best_score = 0
        self.start_time = time.time()
        self.current_time = self.start_time
        self.mps = 15
        self.count = 0

    def is_running(self):
        return self.run

    def stop_running(self):
        self.run = False

    def speedup(self):
        if self.mps < 50:
            self.mps += 1
        self.score = 0
        self.best_score = 0

    def slowdown(self):
        if self.mps > 1:
            self.mps -=1
        self.score = 0
        self.best_score = 0

    def get_mps(self):
        return self.mps

    def reset_grid(self):
        for i in range(100):
            for j in range(100):
                self.grid[i][j] = EMPTY_CHAR
        self.score = 0
        self.best_score = 0

    def expand_row(self):
        if self.rows < 100:
            self.rows += 1
        self.score = 0
        self.best_score = 0

    def expand_column(self):
        if self.columns < 100:
            self.columns += 1
        self.score = 0
        self.best_score = 0

    def shrink_row(self):
        if self.rows > 1:
            self.rows -= 1
        self.score = 0
        self.best_score = 0

    def shrink_column(self):
        if self.columns > 1:
            self.columns -= 1
        self.score = 0
        self.best_score = 0

    def is_alive(self):
        return self.alive

    def remove_food(self):
        if self.food is not None and self.grid[self.food[0]][self.food[1]] == FOOD_CHAR:
            self.grid[self.food[0]][self.food[1]] = EMPTY_CHAR
        self.food = None

    def remove_snake(self):
        for i in range(len(self.snake)):
            pos = self.snake.pop()
            if self.grid[pos[0]][pos[1]] == SNAKE_CHAR:
                self.grid[pos[0]][pos[1]] = EMPTY_CHAR

    def get_available_cells(self):
        available_cells = []
        for i in range(self.rows):
            for j in range(self.columns):
                if self.grid[i][j] == EMPTY_CHAR:
                    available_cells.append((i, j))
        return available_cells

    def get_random_cell(self):
        random_cell = None
        available_cells = self.get_available_cells()
        if len(available_cells) > 0:
            random_cell = random.choice(available_cells)
        return random_cell

    def spawn_snake(self):
        random_cell = self.get_random_cell()
        if random_cell is None:
            self.alive = False
        else:
            self.snake.insert(0, random_cell)
            self.grid[random_cell[0]][random_cell[1]] = SNAKE_CHAR

    def spawn_food(self):
        random_cell = self.get_random_cell()
        if random_cell is None:
            self.alive = False
        else:
            self.grid[random_cell[0]][random_cell[1]] = FOOD_CHAR
            self.food = random_cell

    def start_run(self):
        self.remove_food()
        self.remove_snake()
        self.alive = True
        self.score = 0
        self.previous_move = None
        self.next_move = None
        self.spawn_snake()
        self.spawn_food()
        self.start_time = time.time()
        self.current_time = self.start_time

    def set_next_move(self, move):
        self.next_move = move

    def is_collision(self, pos):
        return not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.columns and self.grid[pos[0]][pos[1]] in [EMPTY_CHAR, FOOD_CHAR])

    def is_next_move_invalid(self):
        if self.previous_move is not None:
           return (self.previous_move[0] + self.next_move[0], self.previous_move[1] + self.next_move[1]) == (0, 0)

    def move_snake(self):
        if self.next_move is None or self.is_next_move_invalid():
            self.next_move = self.previous_move

        if self.next_move is not None:
            head = self.snake[0]
            new_pos = (head[0] + self.next_move[0], head[1] + self.next_move[1])
            if self.is_collision(new_pos):
                self.alive = False
                if self.score > self.best_score:
                    self.best_score = self.score
            else:
                self.snake.insert(0, new_pos)
                self.grid[new_pos[0]][new_pos[1]] = SNAKE_CHAR
                if new_pos == self.food:
                    self.score += 1
                    self.spawn_food()
                else:
                    tail = self.snake.pop()
                    self.grid[tail[0]][tail[1]] = EMPTY_CHAR
                self.previous_move = self.next_move
                self.next_move = None


    def get_state(self):
        return self.grid[:self.rows][:self.columns], self.score, self.alive, self.snake[0], self.food, self.food_eaten()

    def food_eaten(self):
        head = self.snake[0]
        new_pos = (head[0] + self.next_move[0], head[1] + self.next_move[1])
        return new_pos == self.food

    def get_grid_base(self, width, height):
        menu_start = width * 2/3
        vertical_gap = (height - 1) // self.rows
        horizontal_gap = (menu_start - 1) // self.columns
        gap = min(horizontal_gap, vertical_gap)
        vertical_start = (height - self.rows * gap) // 2
        horizontal_start = (menu_start - self.columns * gap) // 2
        return gap, vertical_start, horizontal_start, menu_start

    def get_coord(self, screen, pos):
        gap, vertical_start, horizontal_start, menu_start = self.get_grid_base(screen.get_width(), screen.get_height())
        x, y = pos
        i = int((y - vertical_start) // gap)
        j = int((x - horizontal_start) // gap)
        return i, j

    def add_wall(self, pos):
        i, j = self.get_coord(self.screen, pos)
        if 0 <= i < self.rows and 0 <= j < self.columns:
            self.grid[i][j] = WALL_CHAR
        self.score = 0
        self.best_score = 0

    def remove(self, pos):
        i, j = self.get_coord(self.screen, pos)
        if 0 <= i < self.rows and 0 <= j < self.columns:
            self.grid[i][j] = EMPTY_CHAR
        self.score = 0
        self.best_score = 0

class GUISnakeGame(SnakeGame):
    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 600
    DEFAULT_TITLE_FONT_SIZE = 40
    DEFAULT_FONT_SIZE = 20

    def __init__(self):
        super(GUISnakeGame, self).__init__()
        self.frame = 0

    def next_tick(self, learning_agent=None):
        self.process_event(learning_agent)
        if self.is_alive() and (self.frame / FPS >= 1 / self.get_mps() or learning_agent is not None):
            self.move_snake()
            self.frame = 0
        # drawing on screen
        self.draw()
        self.clock.tick(FPS)
        self.frame += 1

    def process_event(self, learning_agent=None):
        # triggering an event
        for event in pygame.event.get():
            # closing the game
            if event.type == pygame.QUIT:
                self.stop_running()
            elif event.type == pygame.KEYDOWN:
                if not self.is_alive():
                    # start the run
                    if event.key == pygame.K_SPACE:
                        self.start_run()

                    # modify speed
                    elif event.key == pygame.K_u:
                        self.slowdown()
                    elif event.key == pygame.K_i:
                        self.speedup()

                    # modify grid
                    elif event.key == pygame.K_r:
                        self.reset_grid()
                    elif event.key == pygame.K_o:
                        self.expand_row()
                    elif event.key == pygame.K_p:
                        self.expand_column()
                    elif event.key == pygame.K_l:
                        self.shrink_row()
                    elif event.key == pygame.K_SEMICOLON:
                        self.shrink_column()

                if self.is_alive():
                    # controls snake
                    if event.key == pygame.K_UP:
                        self.set_next_move(UP)
                    elif event.key == pygame.K_RIGHT:
                        self.set_next_move(RIGHT)
                    elif event.key == pygame.K_DOWN:
                        self.set_next_move(DOWN)
                    elif event.key == pygame.K_LEFT:
                        self.set_next_move(LEFT)

            # resize window
            elif event.type == pygame.VIDEORESIZE:
                self.set_window_size(event.w, event.h)

            if not self.is_alive():
                # left click
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    self.add_wall(pos)
                # right click
                if pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    self.remove(pos)

        if learning_agent is not None and len(self.snake)>0:
            self.set_next_move(learning_agent.choose_next_move([self.grid, self.snake[0], self.food]))
            learning_agent.update(self.get_state())
            if self.food_eaten():
                self.count = 0
            if not self.is_alive() or self.count > 100:
                learning_agent.replay_new()
                self.start_run()
                self.count = 0
            self.count += 1


    def init_pygame(self):
        pygame.init()
        pygame.font.init()

        self.set_window_size(GUISnakeGame.DEFAULT_WIDTH, GUISnakeGame.DEFAULT_HEIGHT)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

    def set_window_size(self, width, height):
        self.screen = pygame.display.set_mode(size=(width, height), flags=pygame.RESIZABLE)
        ratio = min(width / GUISnakeGame.DEFAULT_WIDTH, height / GUISnakeGame.DEFAULT_HEIGHT)
        self.title_font = pygame.font.Font('./Fonts/Mario-Kart-DS.ttf', round(GUISnakeGame.DEFAULT_TITLE_FONT_SIZE * ratio))
        self.normal_font = pygame.font.Font('./Fonts/Fipps-Regular.otf', round(GUISnakeGame.DEFAULT_FONT_SIZE * ratio))

    def cleanup_pygame(self):
        pygame.font.quit()
        pygame.quit()

    def draw_cells(self, screen, gap, vertical_start, horizontal_start):
        for i in range(self.rows):
            for j in range(self.columns):
                if self.grid[i][j] != EMPTY_CHAR:
                    if self.grid[i][j] == WALL_CHAR:
                        color = WHITE
                    elif self.grid[i][j] == SNAKE_CHAR:
                        color = YELLOW
                    elif self.grid[i][j] == FOOD_CHAR:
                        color = RED
                    pygame.draw.rect(screen, color, (horizontal_start + j * gap, vertical_start + i * gap, gap, gap))

    def draw_grid(self, screen, gap, vertical_start, horizontal_start):
        for i in range(self.rows + 1):
            pygame.draw.line(screen, GREY, (horizontal_start, vertical_start + i * gap), (horizontal_start + self.columns * gap, vertical_start + i * gap), 1)
        for j in range(self.columns + 1):
            pygame.draw.line(screen, GREY, (horizontal_start + j * gap, vertical_start), (horizontal_start + j * gap, vertical_start + self.rows * gap), 1)

    def draw(self):
        self.screen.fill(BLACK)
        width, height = self.screen.get_size()
        gap, vertical_start, horizontal_start, menu_start = self.get_grid_base(width, height)

        # Draw the map
        self.draw_cells(self.screen, gap, vertical_start, horizontal_start)
        self.draw_grid(self.screen, gap, vertical_start, horizontal_start)
        pygame.draw.line(self.screen, GREY, (menu_start, 0), (menu_start, height))

        # Draw texts and timer
        title = self.title_font.render(TITLE, True, WHITE)
        score = self.normal_font.render('Score: ' + str(self.score), True, WHITE)
        highscore = self.normal_font.render('Highscore: ' + str(self.best_score), True, WHITE)
        size = self.normal_font.render('Size: ' + str(self.rows) + 'x' + str(self.columns), True, WHITE)
        mps = self.normal_font.render('MPS: ' + str(self.mps), True, WHITE)
        start = self.normal_font.render('Press Space', True, WHITE)

        if self.alive:
            self.current_time = time.time()

        timer = self.normal_font.render('Timer: ' + str(round(self.current_time - self.start_time, 1)), True, WHITE)
        self.screen.blit(title, (menu_start + (width - menu_start) / 2 - title.get_width() / 2, height * (1/15) - title.get_height()/2))
        self.screen.blit(score, (menu_start + (width - menu_start) / 7, height * (3/15) - score.get_height()/2 ))
        self.screen.blit(highscore, (menu_start + (width - menu_start) / 7, height * (4/15) - highscore.get_height()/2 ))
        self.screen.blit(size, (menu_start + (width - menu_start) / 7, height * (5/15) - size.get_height()/2))
        self.screen.blit(mps, (menu_start + (width - menu_start) / 7, height * (6/15) - mps.get_height()/2))

        if not self.alive:
            self.screen.blit(start, (menu_start + (width - menu_start) / 2 - start.get_width() / 2, height / 2 - start.get_height() / 2))

        self.screen.blit(timer, (menu_start + (width - menu_start) / 7, height - timer.get_height()))
        pygame.display.flip()


class TrainingSnakeGame(SnakeGame):
    def __init__(self, learning_agent):
        super(TrainingSnakeGame, self).__init__()
        self.learning_agent = learning_agent

    def next_tick(self):
        if self.is_alive():
            self.set_next_move(self.learning_agent.choose_next_move(self.get_state()))
            return self.move_snake()

def display_state_console20x20(state):
    grid, score, alive, head = state
    print("Alive: " + str(alive) + " -- Current reward: " + str(score))

    print("  A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T")
    c = ord('A')
    for line in grid[:20]:
        print(" |" + "-+" * 20)
        print(chr(c) + "|" + "|".join(line[:20]))
        c += 1

def main():
    from IA.deep_Q_learning import DQNAgent
    game = GUISnakeGame()
    agent = DQNAgent()  # None for interactive GUI
    agent.to('cpu')
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=agent.learning_rate)

    game.init_pygame()

    # game loop
    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()

    #
    # Training AI example
    #
    #game = TrainingSnakeGame(agent)
    #game.start_run()
    #while game.is_alive():
    #    game.next_tick()

if __name__ == '__main__':
    main()
