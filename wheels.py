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
        self.radius = r
        self.body = None
        self.world = world
        self.on_ground = False
        self.create_wheel_body()
        # Wheels body definition
        body_def = b2BodyDef(type=b2_dynamicBody, position=(x / main.SCALE, y / main.SCALE), angle=0)
        # Fixture wheels
        fix_def = b2FixtureDef(density=0.05, friction=0.99, restitution=0.2,
                               shape=b2CircleShape(radius=(self.radius / main.SCALE)),
                               groupIndex=-1)
        self.rim_body = self.world.CreateBody(body_def)
        self.rim_body.CreateFixture(fix_def)
        self.rim_body.userData = self

        if chassis_body is not None:
            # Create wheel (revolute) joint to car
            rev_joint_def = b2RevoluteJointDef()
            rev_joint_def.Initialize(bodyA=self.body, bodyB=self.rim_body, anchor=self.body.position)
            self.joint = self.world.CreateJoint(rev_joint_def)
            # Create wheel (prismatic) joint to car
            pris_joint_def = b2PrismaticJointDef()
            pris_joint_def.Initialize(bodyA=self.rim_body, bodyB=chassis_body, anchor=self.body.position,
                                      axis=pygame.Vector2(0, -1))
            # Create distance joint between wheel and char
