import math
from Box2D import *
import pygame
import hill_racing


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
                self.starting_position.x / hill_racing.SCALE,
                self.starting_position.y / hill_racing.SCALE,
            ),
            angle=0,
        )
        # Wheel rim fixture
        fix_def = b2FixtureDef(
            density=0.05,
            friction=0.99,
            restitution=0.2,
            shape=b2CircleShape(radius=(self.radius / hill_racing.SCALE)),
            groupIndex=-1,
        )
        self.rim_body = self.world.CreateBody(body_def)
        self.rim_body.CreateFixture(fix_def)
        self.rim_body.userData = self

        if chassis_body is not None:
            # Create wheel (revolute) joint for inner rim and car chassis body
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
            # Create distance joint between wheel and car
            dist_joint_def = b2DistanceJointDef()
            anchor_wheel = b2Vec2(x / hill_racing.SCALE, y / hill_racing.SCALE)
            anchor_car = b2Vec2(x / hill_racing.SCALE, (y - r * 3) / hill_racing.SCALE)
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
                self.starting_position.x / hill_racing.SCALE,
                self.starting_position.y / hill_racing.SCALE,
            ),
            angle=0,
        )
        wheel_fixture = b2FixtureDef(
            density=1,
            friction=1.5,
            restitution=0.1,
            shape=b2CircleShape(radius=self.radius / hill_racing.SCALE),
            categoryBits=hill_racing.WHEEL_CATEGORY,
            maskBits=hill_racing.WHEEL_MASK,
        )
        self.body = self.world.CreateBody(wheel_body)
        self.body.CreateFixture(wheel_fixture)
        self.body.userData = self

    def draw_wheel(self, surface_screen):
        # Scale back position of wheel body
        pos_x = self.body.position.x * hill_racing.SCALE
        pos_y = self.body.position.y * hill_racing.SCALE
        degrees_angle = math.degrees(-self.body.angle) % 360
        # Scale wheel
        hill_racing.wheel_sprite = pygame.transform.scale(
            hill_racing.wheel_sprite, (hill_racing.HEAD_SIZE, hill_racing.HEAD_SIZE)
        )
        # Rotate the wheel by body angle
        rotated_wheel_sprite = pygame.transform.rotate(hill_racing.wheel_sprite, degrees_angle)
        # Update the wheel on screen position
        surface_screen.blit(
            source=rotated_wheel_sprite,
            dest=(-self.radius + pos_x - hill_racing.panX, -self.radius + pos_y - hill_racing.panY),
        )
