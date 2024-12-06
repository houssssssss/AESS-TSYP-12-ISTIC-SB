
import pygame
   import sys
   pygame.init()
   
   WHITE = (255, 255, 255)
   BLUE = (0, 0, 255)
   RED = (255, 0, 0)

   width, height = 800, 600
   screen = pygame.display.set_mode((width, height))
   pygame.display.set_caption('Simulation CanSat')

   cansat_x = width // 2
   cansat_y = height - 50
   velocity = -2 
   parachute_deployed = False

   clock = pygame.time.Clock()

   while True:
       for event in pygame.event.get():
           if event.type == pygame.QUIT:
               pygame.quit()
               sys.exit()

       cansat_y += velocity

Déploiement du parachute à une certaine altitude
       if cansat_y < height // 2 and not parachute_deployed:
           parachute_deployed = True
           velocity = 1  

       screen.fill(WHITE)

Dessin du CanSat
       pygame.draw.rect(screen, BLUE, (cansat_x - 10, cansat_y - 20, 20, 40))

Dessin du parachute si déployé
       if parachute_deployed:
           pygame.draw.polygon(screen, RED, [(cansat_x - 20, cansat_y), (cansat_x + 20, cansat_y), (cansat_x, cansat_y - 40)])

       pygame.display.flip()
       clock.tick(60)