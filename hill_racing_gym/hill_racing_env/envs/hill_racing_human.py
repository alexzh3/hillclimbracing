from gymnasium.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, try 'pip install box2d box2d-kengz'")
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import numpy as np
import math
import ground
import agent

# CHANGE GAMEPLAY AND OTHER VARIABLES IN "hill.racing.py"
# Fundamental constants (not recommended to change)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60  # frames per second
GRAVITY = 10


# Contact listener for head and ground (Bad code)
class ContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact: b2Contact) -> None:
        # Fixture variables
        head_fixture = None
        ground_fixture = None

        # If we contact the head with the ground or vice versa we destroy the car's joints.
        if contact.fixtureA.body.userData.id == "head" and contact.fixtureB.body.userData.id == "ground":
            head_fixture = contact.fixtureA
            ground_fixture = contact.fixtureB
        elif contact.fixtureB.body.userData.id == "head" and contact.fixtureA.body.userData.id == "ground":
            head_fixture = contact.fixtureB
            ground_fixture = contact.fixtureA

        if head_fixture and ground_fixture and head_fixture.body.joints:
            torso = head_fixture.body.joints[0].other  # Get the torso body object using the joint
            car = torso.joints[3].other.userData  # Get the car body using the torso
            car.agent.dead = True

        # Check if we contact the wheel with the ground or vice versa.
        if contact.fixtureA.body.userData.id == "wheel" and contact.fixtureB.body.userData.id == "ground":
            contact.fixtureA.body.userData.on_ground = True
        if contact.fixtureB.body.userData.id == "wheel" and contact.fixtureA.body.userData.id == "ground":
            contact.fixtureB.body.userData.on_ground = True

    def EndContact(self, contact: b2Contact) -> None:
        # End of contact, we need to set the on_ground variable on false
        if contact.fixtureA.body.userData.id == "wheel" and contact.fixtureB.body.userData.id == "ground":
            contact.fixtureA.body.userData.on_ground = False
        if contact.fixtureB.body.userData.id == "wheel" and contact.fixtureA.body.userData.id == "ground":
            contact.fixtureB.body.userData.on_ground = False


# Key events handler when human is playing
def handle_key_events(human_event: pygame.event, human_agent: 'agent.Agent',
                      human_right_down: bool, human_left_down: bool) -> None:
    if human_event.type == pygame.KEYDOWN:
        if human_event.key in (pygame.K_d, pygame.K_RIGHT):
            human_agent.car.motor_on(forward=True)
            human_right_down = True
        elif human_event.key in (pygame.K_a, pygame.K_LEFT):
            human_agent.car.motor_on(forward=False)
            human_left_down = True

    elif human_event.type == pygame.KEYUP:
        if human_event.key in (pygame.K_d, pygame.K_RIGHT):
            human_right_down = False
            if human_left_down:
                human_agent.car.motor_on(forward=False)
            else:
                human_agent.car.motor_off()
        elif human_event.key in (pygame.K_a, pygame.K_LEFT):
            human_left_down = False
            if human_right_down:
                human_agent.car.motor_on(forward=True)
            else:
                human_agent.car.motor_off()


def human_play():
    # Initialize world
    current_ground, current_agent, current_world = setup_world()
    # Initialize key variables for when human plays
    right_key_down = False
    left_key_down = False
    while not current_agent.dead:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:  # Escape to quit game
                print("Escape was pressed, quiting the game...")
                pygame.quit()
            handle_key_events(event, current_agent, right_key_down, left_key_down)

        # Call the draw function
        draw(current_ground, current_agent)
        # Box2D simulation
        current_world.Step(timeStep=1.0 / FPS, velocityIterations=6 * 30, positionIterations=2 * 30)
        # Print for debugging
        print(
            f"position: {current_agent.car.chassis_body.position.x, current_agent.car.chassis_body.position.y},"
            f"wheels_speeds: {current_agent.car.wheels[0].joint.speed, current_agent.car.wheels[1].joint.speed},"
            f"wheels_on_ground: {int(current_agent.car.wheels[0].on_ground), int(current_agent.car.wheels[1].on_ground)},"
            f"FPS: {clock.get_fps()}")
        # Update Agent
        current_agent.update()
        # Update render screen and fps
        pygame.display.flip()
        clock.tick(FPS)
    # Print final distance
    print(f"Final score: {current_agent.score}")
    # Quit the game
    pygame.quit()


def setup_world() -> tuple['ground.Ground', 'agent.Agent', b2World]:
    # Variables
    main_world = b2World(contactListener=ContactListener(), gravity=b2Vec2(0, GRAVITY), doSleep=True)
    ground_template = ground.Ground()  # Template to store the ground vectors
    ground_template.randomize_ground()  # Randomizes the ground using the difficulty and perlin noise

    # Generate until we find ground that is not too steep
    while ground_template.groundTooSteep():
        ground_template = ground.Ground()
        ground_template.randomize_ground()

    # Set up the ground
    main_ground = ground.Ground(main_world)
    main_ground.cloneFrom(ground_template)
    main_ground.setBodies(main_world)

    # Set up the world and agent
    human_agent = agent.Agent(real_world=main_world)
    human_agent.add_to_world()
    return main_ground, human_agent, main_world


def draw(render_ground, render_agent) -> None:
    screen.fill((135, 206, 235))
    # Draw the ground to screen
    render_ground.draw_ground(screen)
    # Draw the agent
    render_agent.draw_agent(screen)
    # Update the screen
    pygame.display.flip()


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Hill climb")
    clock = pygame.time.Clock()
    human_play()
