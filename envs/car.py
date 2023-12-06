import math
import pygame
import random
import hill_racing
import person
import wheels
from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


class Car:
    def __init__(self, x, y, world=None, agent=None):
        self.world = world
        self.id = "car"
        self.agent = agent
        self.wheels = []
        self.starting_position = pygame.Vector2(x, y)
        self.chassis_body = None
        self.chassis_width = 125
        self.chassis_height = 40
        self.wheel_size = 17
        self.dead = False
        self.shapes = []
        self.car_density = 1
        self.car_restitution = 0.01
        self.max_distance = 0
        self.motor_state = 0
        self.rotation_torque = 2
        self.motor_speed = 10

        # vertices for car chassis
        vectors = []
        vectors.append(b2Vec2(-self.chassis_width / 2, 0 - self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 4 + 5, 0 - self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5))
        vectors.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        # Scale vertices
        for vector in vectors:
            vector.x /= hill_racing.SCALE
            vector.y /= hill_racing.SCALE
        self.shapes.append(vectors)

        # Create hill_racing body and fixture for car
        car_body = b2BodyDef(
            type=b2_dynamicBody,
            position=(x / hill_racing.SCALE, y / hill_racing.SCALE),
            angle=0
        )
        car_fixture = b2FixtureDef(
            categoryBits=hill_racing.CHASSIS_CATEGORY,
            maskBits=hill_racing.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors, vertexCount=len(vectors))
        )

        # Create body in world and connect fixture to it
        self.chassis_body = self.world.CreateBody(car_body)
        self.chassis_body.CreateFixture(car_fixture)

        # Create front part car
        vectors2 = []
        vectors2.append(b2Vec2(self.chassis_width / 4, 0 - self.chassis_height / 2))
        vectors2.append(b2Vec2(self.chassis_width / 4 - 15, 0 - self.chassis_height / 2 - 20))
        vectors2.append(b2Vec2(self.chassis_width / 4 - 5, 0 - self.chassis_height / 2 - 20))
        vectors2.append(b2Vec2(self.chassis_width / 4 + 10, 0 - self.chassis_height / 2))
        for vector in vectors2:
            vector.x /= hill_racing.SCALE
            vector.y /= hill_racing.SCALE
        car_fixture2 = b2FixtureDef(
            categoryBits=hill_racing.CHASSIS_CATEGORY,
            maskBits=hill_racing.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors2, vertexCount=len(vectors2))
        )
        self.chassis_body.CreateFixture(car_fixture2)
        self.shapes.append(vectors2)

        # Create back part
        vectors3 = []
        vectors3.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5))
        vectors3.append(b2Vec2(self.chassis_width / 2 + 5, 0 - self.chassis_height / 2 + 8))
        vectors3.append(b2Vec2(self.chassis_width / 2 + 5, self.chassis_height / 2 - 5))
        vectors3.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        for vector in vectors3:
            vector.x /= hill_racing.SCALE
            vector.y /= hill_racing.SCALE

        car_fixture3 = b2FixtureDef(
            categoryBits=hill_racing.CHASSIS_CATEGORY,
            maskBits=hill_racing.CHASSIS_MASK,
            density=self.car_density,
            friction=0.1,
            restitution=0.1,
            shape=b2PolygonShape(vertices=vectors3, vertexCount=len(vectors3))
        )
        self.chassis_body.CreateFixture(car_fixture3)
        self.shapes.append(vectors3)

        # Creating the wheels of the car
        # First the left (back) wheel
        self.wheels.append(
            wheels.Wheel(x - self.chassis_width / 2 + self.wheel_size * 1.2, y + self.chassis_height / 2 +
                         self.wheel_size / 4, self.wheel_size, self.chassis_body, self.world)
        )
        # The right (front) wheel
        self.wheels.append(
            wheels.Wheel(x + self.chassis_width / 2 - self.wheel_size * 1.2, y + self.chassis_height / 2 +
                         self.wheel_size / 4, self.wheel_size, self.chassis_body, self.world)
        )

        # Create the person/character
        self.person = person.Person(x=x, y=y, person_width=hill_racing.PERSON_WIDTH,
                                    person_height=hill_racing.PERSON_HEIGHT, world=self.world)

        # Create revolute joint to connect the torso body to the chassis car body
        rev_joint_def = b2RevoluteJointDef()
        joint_pos = b2Vec2(x / hill_racing.SCALE, y / hill_racing.SCALE)
        rev_joint_def.Initialize(bodyA=self.person.torso.body, bodyB=self.chassis_body, anchor=joint_pos)
        self.rev_joint_torso_chassis = self.world.CreateJoint(rev_joint_def)

        # Create distance joint to connect person's torso and car's chassis
        dist_joint_def = b2DistanceJointDef()
        anchor_person = b2Vec2(x / hill_racing.SCALE, (y - self.person.height * 2 / 3) / hill_racing.SCALE)
        anchor_car = b2Vec2((x + self.chassis_width / 2) / hill_racing.SCALE,
                            (y - self.chassis_height / 2) / hill_racing.SCALE)
        dist_joint_def.Initialize(bodyA=self.person.torso.body, bodyB=self.chassis_body, anchorA=anchor_person,
                                  anchorB=anchor_car)
        dist_joint_def.frequencyHz = 5
        dist_joint_def.dampingRatio = 0.1
        dist_joint_def.length *= 1.1
        self.dist_joint_torso_chassis = self.world.CreateJoint(dist_joint_def)

        # Set chassis_body variables
        self.chassis_body.angularDamping = 0.1
        self.chassis_body.userData = self

    # Function that draws/renders the person, wheels and the car on the screen
    def draw_person_car(self, surface_screen):
        # Get position and angle of the car chassis
        pos_x = self.chassis_body.position.x * hill_racing.SCALE
        pos_y = self.chassis_body.position.y * hill_racing.SCALE
        angle_degree = math.degrees(-self.chassis_body.angle)  # Pygame uses absolute degree, Box2D uses radians
        # Draw person on screen
        self.person.draw_person(surface_screen)
        # Draw wheels on screen
        for wheel in self.wheels:
            wheel.draw_wheel(surface_screen)
        # Scale the car sprite
        hill_racing.car_sprite = pygame.transform.scale(
            hill_racing.car_sprite, (self.chassis_width + 23, self.chassis_height * 2 + 10)
        )
        # Rotate the car and draw the car to screen
        rotated_image = pygame.transform.rotate(hill_racing.car_sprite, angle_degree)
        surface_screen.blit(
            source=rotated_image,
            dest=((-self.chassis_width / 2 - 7) + pos_x - hill_racing.panX,
                  -self.chassis_height - 20 + pos_y - hill_racing.panY)
        )

    # A function that updates whether the agent status is alive or death
    def update_status(self):
        pos_x = self.chassis_body.position.x
        pos_y = self.chassis_body.position.y * hill_racing.SCALE

        # Check whether we are moving forward with the car
        if pos_x > self.max_distance:
            self.max_distance = pos_x

        # When agent is out of the screen height, we set status to dead
        if not self.dead and pos_y > hill_racing.SCREEN_HEIGHT:
            self.dead = True
            self.agent.dead = True

    # Function that turns on the motor on wheels and moves forward
    def motor_on(self, forward: bool):
        self.wheels[0].joint.motorEnabled = True
        self.wheels[1].joint.motorEnabled = True
        old_state = self.motor_state
        if forward:  # When we move forward / give gas
            self.motor_state = 1
            self.wheels[0].joint.motorSpeed = -self.motor_speed * math.pi
            self.wheels[1].joint.motorSpeed = -self.motor_speed * math.pi
            self.chassis_body.ApplyTorque(-self.rotation_torque, True)
        else:  # Reverse, gas to the other side
            self.motor_state = -1
            self.wheels[0].joint.motorSpeed = self.motor_speed * math.pi
            self.wheels[1].joint.motorSpeed = self.motor_speed * math.pi
        # Rotation applied to the car when we stop giving gas
        if old_state + self.motor_state == 0:
            if old_state == 1:
                self.chassis_body.ApplyTorque(self.motor_state * -1, True)

        # Set maximum motor torque on wheels
        self.wheels[0].joint.maxMotorTorque = 500
        self.wheels[1].joint.maxMotorTorque = 250

    # When we brake, we turned the motor off, that will also apply torque
    def motor_off(self):
        if self.motor_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * self.rotation_torque, True)
        self.motor_state = 0
        self.wheels[0].joint.motorEnabled = False
        self.wheels[1].joint.motorEnabled = False
