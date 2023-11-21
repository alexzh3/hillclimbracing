import math

from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")
import pygame
import main


class Person:
    def __init__(self, x, y, person_width, person_height, world=None):
        self.world = world
        self.x = x
        self.y = y
        self.height = person_height
        self.width = person_width
        self.torso = Torso(x, y - (person_height / 2), person_height, person_width, self.world)
        self.head = Head(x, y - (person_height + person_width * 1), person_width * 1, self.world)

        # Create revolute (angle) joint for person body and head
        rev_joint_def = b2RevoluteJointDef()
        joint_angle = [x / main.SCALE, (y - person_height) / main.SCALE]
        rev_joint_def.Initialize(bodyA=self.head.body, bodyB=self.torso.body, anchor=joint_angle)
        self.rev_joint_head_torso = self.world.CreateJoint(rev_joint_def)

        # Create distance joint for head and torso
        dist_joint_def = b2DistanceJointDef()
        anchor_torso = [x / main.SCALE, y / main.SCALE]
        anchor_head = [x / main.SCALE, self.head.starting_position.y / main.SCALE]
        dist_joint_def.Initialize(
            bodyA=self.head.body,
            bodyB=self.torso.body,
            anchorA=anchor_head,
            anchorB=anchor_torso
        )
        self.dist_joint_head_torso = self.world.CreateJoint(dist_joint_def)

    # Function to render/draw the head and torso
    def draw_person(self):
        self.head.draw_head()
        self.torso.draw_torso()


class Head:
    def __init__(self, x, y, r, world=None):
        self.world = world
        self.starting_position = pygame.Vector2(x, y)
        self.radius = r
        self.body = None
        self.id = "head"
        self.is_CB = False
        self.make_head_body()

    # Function that creates the body of the head
    def make_head_body(self):
        body_def = b2BodyDef()
        body_def.type = b2_dynamicBody

        body_def.position.x = self.starting_position.x / main.SCALE
        body_def.position.y = self.starting_position.y / main.SCALE
        body_def.angle = 0
        fix_def = b2FixtureDef(
            categoryBits=main.PERSON_CATEGORY,
            maskBits=main.PERSON_MASK,
            density=0.001,
            friction=0.01,
            restitution=0.01,
            shape=b2CircleShape(radius=self.radius / main.SCALE)
        )
        self.body = self.world.CreateBody(body_def)
        self.body.userData = self
        self.body.CreateFixture(fix_def)

    # Function that draws the head
    def draw_head(self):
        pos_x = self.body.position.x * main.SCALE
        pos_y = self.body.position.y * main.SCALE
        degrees_angle = math.degrees(self.body.angle)
        # Scale head sprite
        main.head_sprite = pygame.transform.scale(
            main.head_sprite, (main.WHEEL_SIZE, main.WHEEL_SIZE)
        )
        # Get angle and rotate head
        rotated_head_sprite = pygame.transform.rotate(main.head_sprite, degrees_angle)
        # Update the head on screen position
        main.screen.blit(
            source=rotated_head_sprite,
            dest=(pos_x - main.panX - self.radius + 12, pos_y - main.panY - self.radius + 18)
        )


class Torso:
    def __init__(self, center_x, center_y, height, width, world):
        self.id = "torso"
        self.world = world
        self.width = width
        self.height = height
        self.starting_position = pygame.Vector2(center_x, center_y)
        self.body = None
        self.make_torso_body()

    # Function that creates the torso body of the person
    def make_torso_body(self):
        body_def = b2BodyDef()
        body_def.type = b2_dynamicBody
        body_def.position.x = self.starting_position.x / main.SCALE
        body_def.position.y = self.starting_position.y / main.SCALE
        body_def.angle = 0

        fix_def = b2FixtureDef(
            categoryBits=main.PERSON_CATEGORY,
            maskBits=main.PERSON_MASK,
            density=0.002,
            friction=0.01,
            restitution=0.01,
            shape=b2PolygonShape()
        )
        fix_def.shape.SetAsBox(self.width / 2 / main.SCALE, self.height / main.SCALE)

        self.body = self.world.CreateBody(body_def)
        self.body.userData = self
        self.body.CreateFixture(fix_def)

    # Function that draws the torso to the screen
    def draw_torso(self):
        pos_x = self.body.position.x * main.SCALE
        pos_y = self.body.position.y * main.SCALE
        degrees_angle = abs(math.degrees(self.body.angle))

        main.torso_sprite = pygame.transform.scale(
            main.torso_sprite, (main.PERSON_WIDTH, main.PERSON_HEIGHT)
        )
        # Get angle and rotate head
        rotated_torso_sprite = pygame.transform.rotate(main.torso_sprite, degrees_angle)
        # Update the head on screen position
        main.screen.blit(
            source=rotated_torso_sprite,
            dest=(pos_x - main.panX, pos_y - main.panY)
        )
