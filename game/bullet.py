import pygame
import math
from collections import deque
from configs.game_config import (SCREEN_WIDTH, SCREEN_HEIGHT, USE_BULLET_COLORS,
                      DISPLAY_BULLET_TRAIL, TRAIL_MAX_LENGTH, UPDATE_DELTA_TIME)
from utils.draw_utils import draw_water_drop
from configs.game_config import DynamicConfig

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, angle: int, speed: int, radius: int, color, fade=0, bouncing=False, from_corner=False, max_bounces=5):
        super().__init__()
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.fade = fade
        self.origin_color = color if DynamicConfig.USE_BULLET_COLORS else (255, 255, 255)
        self.color = self.origin_color
        self.bouncing = bouncing
        self.from_corner = from_corner
        self.radius = radius
        self.alpha = 255 if fade else None
        self.spawn_event = []
        self.trail = deque(maxlen=TRAIL_MAX_LENGTH) if DISPLAY_BULLET_TRAIL else None
        self.bounce_count = 0
        self.max_bounces = max_bounces
        self.was_bouncing = False 
        

        

    def add_spawn_event(self, bullet_type, spawn_time, delay):
        current_time = pygame.time.get_ticks()
        self.spawn_event.append({
            'bullet_type': bullet_type,
            'spawn_time': spawn_time,
            'delay': current_time + delay})    
        
    def update(self):
        # inefficient due to constantly re-calculating sine and cosine
        self.x += math.cos(self.angle) * self.speed * UPDATE_DELTA_TIME
        self.y += math.sin(self.angle) * self.speed * UPDATE_DELTA_TIME
        
        if self.bouncing:
            bounced = False
            if self.x - self.radius <= 0 or self.x + self.radius >= SCREEN_WIDTH:
                self.angle = math.pi - self.angle  
                self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))
                bounced = True
            if self.y - self.radius <= 0 or self.y + self.radius >= SCREEN_HEIGHT:
                self.angle = - self.angle  
                self.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.y))
                bounced = True

            if bounced and not self.was_bouncing:
                self.bounce_count += 1
                self.was_bouncing = True
            elif not bounced:
                self.was_bouncing = False

            if self.bounce_count >= self.max_bounces:
                self.kill()

        if self.fade:
            self.alpha = max(0, self.alpha - self.fade)
        if self.is_out_of_bounds():
            self.kill()

    def draw(self, surface):
        if DynamicConfig.DISPLAY_BULLET_TRAIL and self.trail is not None:
            self.trail.append((self.x, self.y))
            draw_water_drop(surface, self)
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

        
    def is_out_of_bounds(self):
        return not (0 <= self.x <= SCREEN_WIDTH and 0 <= self.y <= SCREEN_HEIGHT)
    
    def set_color(self, color):
        self.color = color