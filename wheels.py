import math

from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")
import pygame
import main


class Wheel:
    def __init__(self, x, y, r, chassis_body=None, world=None):
        self.starting_position = pygame.Vector2(x, y)
        self.id = "wheel"
        self.radius = r
        self.body = None
        self.world = world
        self.on_ground = False
        # Create wheel
        self.create_wheel()
        # Wheel rim body definition
        body_def = b2BodyDef(
            type=b2_dynamicBody,
            position=(
                self.starting_position.x / main.SCALE,
                self.starting_position.y / main.SCALE,
            ),
            angle=0,
        )
        # Wheel rim fixture
        fix_def = b2FixtureDef(
            density=0.05,
            friction=0.99,
            restitution=0.2,
            shape=b2CircleShape(radius=(self.radius / main.SCALE)),
            groupIndex=-1,
        )
        self.rim_body = self.world.CreateBody(body_def)
        self.rim_body.CreateFixture(fix_def)
        self.rim_body.userData = self

        if chassis_body is not None:
            # Create wheel (revolute) joint for inner rim and wheel outer
            rev_joint_def = b2RevoluteJointDef()
            rev_joint_def.Initialize(
                bodyA=self.body, bodyB=self.rim_body, anchor=self.body.position
            )
            self.joint = self.world.CreateJoint(rev_joint_def)
            # Create wheel (prismatic) joint to car chassis
            pris_joint_def = b2PrismaticJointDef()
            pris_joint_def.Initialize(
                bodyA=self.rim_body,
                bodyB=chassis_body,
                anchor=self.body.position,
                axis=b2Vec2(0, -1),
            )
            self.pris_joint = self.world.CreateJoint(pris_joint_def)
            # Create distance joint between wheel and char
            dist_joint_def = b2DistanceJointDef()
            anchor_wheel = b2Vec2(x / main.SCALE, y / main.SCALE)
            anchor_car = b2Vec2(x / main.SCALE, (y - r * 3) / main.SCALE)
            dist_joint_def.Initialize(
                bodyA=self.rim_body,
                bodyB=chassis_body,
                anchorA=anchor_wheel,
                anchorB=anchor_car,
            )
            dist_joint_def.frequencyHz = 70
            dist_joint_def.dampingRatio = 25
            self.dist_joint = self.world.CreateJoint(dist_joint_def)
        self.body.angularDamping = 1.8

    def create_wheel(self):
        wheel_body = b2BodyDef(
            type=b2_dynamicBody,
            position=(
                self.starting_position.x / main.SCALE,
                self.starting_position.y / main.SCALE,
            ),
            angle=0,
        )
        wheel_fixture = b2FixtureDef(
            density=1,
            friction=1.5,
            restitution=0.1,
            shape=b2CircleShape(radius=self.radius / main.SCALE),
            categoryBits=main.WHEEL_CATEGORY,
            maskBits=main.WHEEL_MASK,
        )
        self.body = self.world.CreateBody(wheel_body)
        self.body.CreateFixture(wheel_fixture)
        self.body.userData = self

    def draw_wheel(self):
        # Scale back position of wheel body
        pos_x = self.body.position.x * main.SCALE
        pos_y = self.body.position.y * main.SCALE
        degrees_angle = math.degrees(self.body.angle)
        # Scale wheel
        main.wheel_sprite = pygame.transform.scale(
            main.wheel_sprite, (main.WHEEL_SIZE * 2, main.WHEEL_SIZE * 2)
        )
        # Rotate the wheel by body angle
        # main.wheel_sprite = pygame.transform.rotate(main.wheel_sprite, degrees_angle)
        # Update the wheel on screen position
        main.screen.blit(
            source=main.wheel_sprite,
            dest=(-self.radius + pos_x - main.panX, -self.radius + pos_y - main.panY),
        )
