#!/usr/bin/env python3
from typing import List, Set
from dataclasses import dataclass
import pygame
from enum import Enum, unique
import sys
import random

import time
import heapq as hq


FPS = 12

INIT_LENGTH = 4

WIDTH = 480
HEIGHT = 480
GRID_SIDE = 24
GRID_WIDTH = WIDTH // GRID_SIDE
GRID_HEIGHT = HEIGHT // GRID_SIDE

BRIGHT_BG = (103, 223, 235)
DARK_BG = (78, 165, 173)
GREEN = (0, 255, 0)

SNAKE_COL = (6, 38, 7)
FOOD_COL = (224, 160, 38)
OBSTACLE_COL = (209, 59, 59)
VISITED_COL = (24, 42, 142)


@unique
class Direction(tuple, Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def reverse(self):
        x, y = self.value
        return Direction((x * -1, y * -1))


@dataclass
class Position:
    x: int
    y: int

    def check_bounds(self, width: int, height: int):
        return (self.x >= width) or (self.x < 0) or (self.y >= height) or (self.y < 0)

    def draw_node(self, surface: pygame.Surface, color: tuple, background: tuple):
        r = pygame.Rect(
            (int(self.x * GRID_SIDE), int(self.y * GRID_SIDE)), (GRID_SIDE, GRID_SIDE)
        )
        pygame.draw.rect(surface, color, r)
        pygame.draw.rect(surface, background, r, 1)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Position):
            return (self.x == o.x) and (self.y == o.y)
        else:
            return False

    def __str__(self):
        return f"X{self.x};Y{self.y};"

    def __hash__(self):
        return hash(str(self))


class GameNode:
    nodes: Set[Position] = set()

    def __init__(self):
        self.position = Position(0, 0)
        self.color = (0, 0, 0)

    def randomize_position(self):
        try:
            GameNode.nodes.remove(self.position)
        except KeyError:
            pass

        condidate_position = Position(
            random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1),
        )

        if condidate_position not in GameNode.nodes:
            self.position = condidate_position
            GameNode.nodes.add(self.position)
        else:
            self.randomize_position()

    def draw(self, surface: pygame.Surface):
        self.position.draw_node(surface, self.color, BRIGHT_BG)


class Food(GameNode):
    def __init__(self):
        super(Food, self).__init__()
        self.color = FOOD_COL
        self.randomize_position()


class Obstacle(GameNode):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.color = OBSTACLE_COL
        self.randomize_position()


class Snake:
    def __init__(self, screen_width, screen_height, init_length):
        self.color = SNAKE_COL
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.init_length = init_length
        self.reset()

    def reset(self):
        self.length = self.init_length
        self.positions = [Position((GRID_SIDE / 2), (GRID_SIDE / 2))]
        self.direction = random.choice([e for e in Direction])
        self.score = 0

    def get_head_position(self) -> Position:
        return self.positions[0]

    def turn(self, direction: Direction):
        if self.length > 1 and direction.reverse() == self.direction:
            return
        else:
            self.direction = direction

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction.value
        new = Position(cur.x + x, cur.y + y,)
        if self.collide(new):
            self.reset()
        else:
            self.positions.insert(0, new)
            while len(self.positions) > self.length:
                self.positions.pop()

    def collide(self, new: Position):
        return (new in self.positions) or (new.check_bounds(GRID_WIDTH, GRID_HEIGHT))

    def eat(self, food: Food):
        if self.get_head_position() == food.position:
            self.length += 1
            self.score += 1
            food.randomize_position()

    def hit_obstacle(self, obstacle: Obstacle):
        if self.get_head_position() == obstacle.position:
            self.length -= 1
            self.score -= 1
            if self.length == 0:
                self.reset()

    def draw(self, surface: pygame.Surface):
        for p in self.positions:
            p.draw_node(surface, self.color, BRIGHT_BG)


class Player:
    def __init__(self) -> None:
        self.visited_color = VISITED_COL
        self.visited: Set[Position] = set()
        self.chosen_path: List[Direction] = []

    def move(self, snake: Snake) -> bool:
        try:
            next_step = self.chosen_path.pop(0)
            snake.turn(next_step)
            return False
        except IndexError:
            return True

    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def turn(self, direction: Direction):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def draw_visited(self, surface: pygame.Surface):
        for p in self.visited:
            p.draw_node(surface, self.visited_color, BRIGHT_BG)


class SnakeGame:
    def __init__(self, snake: Snake, player: Player) -> None:
        pygame.init()

        self.snake = snake
        self.food = Food()
        self.obstacles: Set[Obstacle] = set()
        for _ in range(40):
            ob = Obstacle()
            while any([ob.position == o.position for o in self.obstacles]):
                ob.randomize_position()
            self.obstacles.add(ob)

        self.player = player

        self.fps_clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(
            (snake.screen_height, snake.screen_width), 0, 32
        )
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.myfont = pygame.font.SysFont("monospace", 16)

    def drawGrid(self):
        for y in range(0, int(GRID_HEIGHT)):
            for x in range(0, int(GRID_WIDTH)):
                p = Position(x, y)
                if (x + y) % 2 == 0:
                    p.draw_node(self.surface, BRIGHT_BG, BRIGHT_BG)
                else:
                    p.draw_node(self.surface, DARK_BG, DARK_BG)

    def run(self):
        while not self.handle_events():
            self.fps_clock.tick(FPS)
            self.drawGrid()
            if self.player.move(self.snake):
                self.player.search_path(self.snake, self.food, self.obstacles)
                self.player.move(self.snake)
            self.snake.move()
            self.snake.eat(self.food)
            for ob in self.obstacles:
                self.snake.hit_obstacle(ob)
            for ob in self.obstacles:
                ob.draw(self.surface)
            self.player.draw_visited(self.surface)
            self.snake.draw(self.surface)
            self.food.draw(self.surface)
            self.screen.blit(self.surface, (0, 0))
            text = self.myfont.render(
                "Score {0}".format(self.snake.score), 1, (0, 0, 0)
            )
            self.screen.blit(text, (5, 10))
            pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_UP:
                    self.player.turn(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self.player.turn(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.player.turn(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.player.turn(Direction.RIGHT)
        return False


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()

    def turn(self, direction: Direction):
        self.chosen_path.append(direction)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

@dataclass
class State(Position):
    predecessor: None
    moveFromPredecessor: Direction
    cost: int = 0
    entry: int = 0
    def __gt__(self, o):
        if(self.cost == o.cost):
            return self.entry > o.entry #tiebreak
        return self.cost > o.cost
    def __eq__(self, o):
        return super().__eq__(o)
    def __repr__(self):
        predecessor = None
        if(self.predecessor is not None):
            predecessor = f"X{self.x};Y{self.y};";
        return "{} predecessor: {} moveFromPredecessor: {} entry: {} value: {}".format(super().__repr__(),predecessor,self.moveFromPredecessor,self.entry,self.cost)
    def __hash__(self):
        return hash(str(self))

class SearchBasedPlayer(Player):
    concidered:List[State] = []
    expanded:List[State] = []

    def __init__(self,algorithm):
        super(SearchBasedPlayer, self).__init__()
        self.counter = 0
        self.algorithm = algorithm

    def expand(self,state: State,snake: Snake,obstacles:Set[Obstacle],goal: State):
        for direction in Direction:
            self.counter += 1
            concideredState = State(state.x+direction[0],state.y+direction[1],state,direction,state.cost, self.counter)
            #check Boundaries
            if(not concideredState.check_bounds(GRID_WIDTH,GRID_HEIGHT)):                
                #check concidered
                if(concideredState not in self.concidered):
                    #check expanded
                    if(concideredState not in self.expanded):
                        #check snakeBody
                        if(concideredState not in snake.positions):
                            if(self.algorithm == "BFS"):
                                self.concidered.append(concideredState)
                            elif(self.algorithm == "DFS"):
                                self.concidered.insert(0,concideredState)
                            elif(self.algorithm == "Heuristic"):
                                concideredState.cost += (abs(goal.x-concideredState.x) + abs(goal.y-concideredState.y))
                                hq.heappush(self.concidered,concideredState)
                            else:#A-Star
                                #Dijkstra
                                for ob in obstacles:
                                    if ob.position == Position(concideredState.x,concideredState.y):
                                        concideredState.cost += 50
                                        break;
                                #heuristic
                                concideredState.cost += (abs(goal.x-concideredState.x) + abs(goal.y-concideredState.y))
                                hq.heappush(self.concidered,concideredState)

    def search_path(self, snake: Snake, food: Food, obstacles: Set[Obstacle]):
        #resetHistory
        self.visited.clear()
        self.concidered = []
        self.expanded = []
        goal = State(food.position.x,food.position.y,None,None,0,0)
        #BFS/DFS
        start = State(snake.get_head_position().x,snake.get_head_position().y,None,None,0, self.counter)
        self.concidered = [start]
        #search-Loop
        while True:
            #expand
            if(self.algorithm == "DFS" or self.algorithm =="BFS"):
                currentNode = self.concidered.pop(0)            
            else:
                #Heuristic/A-Star
                currentNode = hq.heappop(self.concidered)
            self.expanded.append(currentNode)
            #expand and validate
            self.expand(currentNode,snake,obstacles,goal)
            self.visited.add(currentNode)
            #check for no further steps
            if(len(self.concidered) == 0):
                print("no further Steps possible")   
                time.sleep(10)
                #concider obsticles 
                break
            #check for food
            if(goal in self.concidered):
                #print("found path")
                index = self.concidered.index(goal)
                iState:State = self.concidered[index]
                while iState is not start:
                    self.chosen_path. append(iState.moveFromPredecessor)
                    #print(iState.moveFromPredecessor)
                    iState = iState.predecessor
                self.chosen_path.reverse();
                break


if __name__ == "__main__":
    snake = Snake(WIDTH, WIDTH, INIT_LENGTH)
    #player = HumanPlayer()
    algorithm = ["BFS","DFS","Heuristic","AStar"]
    player = SearchBasedPlayer(algorithm[2])
    game = SnakeGame(snake, player)
    game.run()


    #question how to handel class in itself without None?
    #why did he put a * infront of obsticles parameter