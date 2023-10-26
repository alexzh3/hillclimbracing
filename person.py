from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")
import pygame
import main


class Person:
    def __init__(self, x, y, person_width, person_height, screen=None, world=None):
        self.world = world
        self.screen = screen
        self.x = x
        self.y = y
        self.height = person_height
        self.width = person_width
        self.torso = Torso(x, y - (person_height / 2), person_width, self.world)
        self.head = Head(x, y - (person_height + person_width * 1), person_width * 1, self.world)

        # Create revolute (angle) joint for person body and head
        rev_joint_def = b2RevoluteJointDef()
        joint_angle = [x / main.SCALE, (y - person_height) / main.SCALE]
        rev_joint_def.Initialize(bodyA=self.head.body, bodyB=self.torso.body, anchor=joint_angle)
        self.head_joint = self.world.CreateJoint(rev_joint_def)

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
        self.dist_joint = self.world.CreateJoint(dist_joint_def)

    # Function to render/draw the head and torso
    def draw_person(self):
        self.head.draw_head(self.screen)
        self.torso.draw_torso(self.screen)


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
            shape=b2CircleShape(self.radius / main.SCALE)
        )
        self.body = self.world.CreateBody(body_def)
        self.body.SetUserData = self
        self.body.CreateFixture(fix_def)

    # Function that draws the head
    def draw_head(self, screen):
        # x = self.body.GetPosition().x * main.SCALE
        # y = self.body.GetPosition().y * main.SCALE
        # Get angle and rotate head
        angle = self.body.GetAngle()
        main.head_sprite = pygame.transform.rotate(main.head_sprite, angle)
        # Update the head on screen position
        screen.blit(
            source=main.head_sprite,
            dest=(-self.radius - 8, -self.radius - 15)
        )


class Torso:
    ...
