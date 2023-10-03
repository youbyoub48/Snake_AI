#Snake game by https://www.geeksforgeeks.org/snake-game-in-python-using-pygame-module/

import pygame
import time
import random

from PIL import Image

 
class Game():
    def __init__(self):
        self.run_game()
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
    
    # game over function
    def game_over(self):
    
        # creating font object my_font
        my_font = pygame.font.SysFont('times new roman', 50)
        
        # creating a text surface on which text
        # will be drawn
        game_over_surface = my_font.render(
            'Your Score is : ' + str(self.score), True, self.red)
        
        # create a rectangular object for the text
        # surface object
        game_over_rect = game_over_surface.get_rect()
        
        # setting position of the text
        game_over_rect.midtop = (self.window_x/2, self.window_y/4)
        
        # blit will draw the text on screen
        self.game_window.blit(game_over_surface, game_over_rect)
        pygame.display.flip()
        
        # after 2 seconds we will quit the program
        time.sleep(2)
        
        # deactivating pygame library
        pygame.quit()
        
        # quit the program
        quit()
    

    def run_game(self):
        snake_speed = 15
    
        # Window size
        self.window_x = 720
        self.window_y = 480
        
        # defining colors
        black = pygame.Color(0, 0, 0)
        white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        green = pygame.Color(0, 255, 0)
        blue = pygame.Color(0, 0, 255)
        
        # Initialising pygame
        pygame.init()
        
        # Initialise game window
        pygame.display.set_caption('GeeksforGeeks Snakes')
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
            if change_to == 'UP' and direction != 'DOWN':
                direction = 'UP'
            if change_to == 'DOWN' and direction != 'UP':
                direction = 'DOWN'
            if change_to == 'LEFT' and direction != 'RIGHT':
                direction = 'LEFT'
            if change_to == 'RIGHT' and direction != 'LEFT':
                direction = 'RIGHT'

            pygame.image.save(self.game_window,"screenshot.jpg")
            img = Image.open("screenshot.jpg")

            img = img.convert("L")
            img.save("mono.jpg")
        
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
            pygame.draw.rect(self.game_window, self.red, pygame.Rect(
                fruit_position[0], fruit_position[1], 10, 10))
        
            # Game Over conditions
            if snake_position[0] < 0 or snake_position[0] > self.window_x-10:
                self.game_over()
            if snake_position[1] < 0 or snake_position[1] > self.window_y-10:
                self.game_over()
        
            # Touching the snake body
            for block in snake_body[1:]:
                if snake_position[0] == block[0] and snake_position[1] == block[1]:
                    self.game_over()
        
            # displaying score continuously
            self.show_score(1, white, 'times new roman', 20)
        
            # Refresh game screen
            pygame.display.update()
        
            # Frame Per Second /Refresh Rate
            fps.tick(snake_speed)


if __name__ == "__main__":
    game = Game()