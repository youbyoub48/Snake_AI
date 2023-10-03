#Snake game by https://www.geeksforgeeks.org/snake-game-in-python-using-pygame-module/

import pygame
import time
import random
import neat
import os

from PIL import Image

 
class Game():
    def __init__(self,genome,config):
        self.run_game(genome,config)
    # displaying Score function
    def show_score(self,choice, color, font, size):
    
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)
        
        # create the display surface object
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        
        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        
        # displaying text
        self.game_window.blit(score_surface, score_rect)
    
    

    def run_game(self,genome,config):
        snake_speed = 15
    
        # Window size
        self.window_x = 420
        self.window_y = 380
        
        # defining colors
        black = pygame.Color(0, 0, 0)
        white = pygame.Color(255, 255, 255)
        red = pygame.Color(255, 0, 0)
        green = pygame.Color(0, 255, 0)
        blue = pygame.Color(0, 0, 255)
        
        # Initialising pygame
        pygame.init()
        
        # Initialise game window
        pygame.display.set_caption('Snake AI')
        self.game_window = pygame.display.set_mode((self.window_x, self.window_y))
        
        # FPS (frames per second) controller
        fps = pygame.time.Clock()
        
        # defining snake default position
        snake_position = [100, 50]
        
        # defining first 4 blocks of snake body
        snake_body = [[100, 50],
                    [90, 50],
                    [80, 50],
                    [70, 50]
                    ]
        # fruit position
        fruit_position = [random.randrange(1, (self.window_x//10)) * 10,
                        random.randrange(1, (self.window_y//10)) * 10]
        
        fruit_spawn = True
        
        # setting default snake direction towards
        # right
        direction = 'RIGHT'
        change_to = direction
        
        # initial score
        self.score = 0

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        while True:
            
            # handling key events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        change_to = 'RIGHT'
        
            # If two keys pressed simultaneously
            # we don't want snake to move into two
            # directions simultaneously
            """
            if change_to == 'UP' and direction != 'DOWN':
                direction = 'UP'
            if change_to == 'DOWN' and direction != 'UP':
                direction = 'DOWN'
            if change_to == 'LEFT' and direction != 'RIGHT':
                direction = 'LEFT'
            if change_to == 'RIGHT' and direction != 'LEFT':
                direction = 'RIGHT'
            """

            pygame.image.save(self.game_window,"screenshot.jpg")
            img = Image.open("screenshot.jpg")

            img = img.convert("L")

            pixels = []

            for y in range(self.window_y):
                for x in range(self.window_x):
                    pixels.append(img.getpixel((x,y)))
            
            output = net.activate(pixels)

            decision = output.index(max(output))

            if decision == 0 and direction != 'DOWN':
                direction = 'UP'
            if decision == 1 and direction != 'UP':
                direction = 'DOWN'
            if decision == 2 and direction != 'RIGHT':
                direction = 'LEFT'
            if decision == 3 and direction != 'LEFT':
                direction = 'RIGHT'
        
            # Moving the snake
            if direction == 'UP':
                snake_position[1] -= 10
            if direction == 'DOWN':
                snake_position[1] += 10
            if direction == 'LEFT':
                snake_position[0] -= 10
            if direction == 'RIGHT':
                snake_position[0] += 10
        
            # Snake body growing mechanism
            # if fruits and snakes collide then scores
            # will be incremented by 10
            snake_body.insert(0, list(snake_position))
            if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
                self.score += 10
                fruit_spawn = False
            else:
                snake_body.pop()
                
            if not fruit_spawn:
                fruit_position = [random.randrange(1, (self.window_x//10)) * 10,
                                random.randrange(1, (self.window_y//10)) * 10]
                
            fruit_spawn = True
            self.game_window.fill(blue)
            
            for pos in snake_body:
                pygame.draw.rect(self.game_window, green,
                                pygame.Rect(pos[0], pos[1], 10, 10))
            pygame.draw.rect(self.game_window, red, pygame.Rect(
                fruit_position[0], fruit_position[1], 10, 10))
        
            # Game Over conditions
            if snake_position[0] < 0 or snake_position[0] > self.window_x-10:
                genome.fitness = self.score
                pygame.quit()
                break
            if snake_position[1] < 0 or snake_position[1] > self.window_y-10:
                genome.fitness = self.score
                pygame.quit()
                break
        
            # Touching the snake body
            for block in snake_body[1:]:
                if snake_position[0] == block[0] and snake_position[1] == block[1]:
                    genome.fitness = self.score
                    pygame.quit()
                    break
        
            # displaying score continuously
            self.show_score(1, white, 'times new roman', 20)
        
            # Refresh game screen
            pygame.display.update()
        
            # Frame Per Second /Refresh Rate
            fps.tick(snake_speed)

def eval_genomes(genomes,config):
    for genome_id, genome in genomes:
        game = Game(genome,config)

def run_ai(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run_ai(config_path)