import random, math, pygame
import main, wheels, person
from gym.error import DependencyNotInstalled

try:
    from Box2D import *
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")


class Car:
    def __init__(self, x, y, world=None, player=None):
        self.world = world
        self.player = player
        self.wheels = []
        self.starting_position = pygame.Vector2(x, y)
        self.chassis_body = None
        self.chassis_width = 125
        self.chassis_height = 40
        self.wheel_size = 17
        self.dead = False
        self.change_counter = 0
        self.shapes = []
        self.car_density = 1
        self.car_restitution = 0.01
        self.max_distance = 0
        self.motor_state = 0
        self.rotation_torque = 2
        self.motor_speed = 13

        # vertices for car chassis
        vectors = []
        vectors.append(b2Vec2(-self.chassis_width / 2, 0 - self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 4 + 5, 0 - self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5))
        vectors.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        vectors.append(b2Vec2(self.chassis_width / 2, self.chassis_height / 2))
        self.shapes.append(vectors)

        # Scale vertices
        for vector in vectors:
            vector.x /= main.SCALE
            vector.y /= main.SCALE

        # Create main body and fixture for car
        car_body = b2BodyDef(
            type=b2_dynamicBody,
            position=(x / main.SCALE / y / main.SCALE),
            angle=0
        )
        car_fixture = b2FixtureDef(
            categoryBits=main.CHASSIS_CATEGORY,
            maskBits=main.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors)
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
        self.shapes.append(vectors2)

        for vector in vectors2:
            vector.x /= main.SCALE
            vector.y /= main.SCALE

        car_fixture2 = b2FixtureDef(
            categoryBits=main.CHASSIS_CATEGORY,
            maskBits=main.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors2)
        )
        self.chassis_body.CreateFixture(car_fixture2)

        # Create back part
        vectors3 = []
        vectors3.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5))
        vectors3.append(b2Vec2(self.chassis_width / 2 + 5, 0 - self.chassis_height / 2 + 8))
        vectors3.append(b2Vec2(self.chassis_width / 2 + 5, 0 - self.chassis_height / 2 - 5))
        vectors3.append(b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2))
        for vector in vectors3:
            vector.x /= main.SCALE
            vector.y /= main.SCALE

        car_fixture3 = b2FixtureDef(
            categoryBits=main.CHASSIS_CATEGORY,
            maskBits=main.CHASSIS_MASK,
            density=self.car_density,
            friction=0.1,
            restitution=0.1,
            shape=b2PolygonShape(vertices=vectors3)
        )
        self.chassis_body.CreateFixture(car_fixture3)
        self.shapes.append(vectors3)

        # Creating the wheels of the car
        # First the left wheel
        self.wheels.append(
            wheels.Wheel(x - self.chassis_width / 2 + self.wheel_size * 1.2, y + self.chassis_height / 2 +
                         self.wheel_size / 4, self.wheel_size, self.chassis_body, self.world)
        )
        # The right wheel
        self.wheels.append(
            wheels.Wheel(x + self.chassis_width / 2 - self.wheel_size * 1.2, y + self.chassis_height / 2 +
                         self.wheel_size / 4, self.wheel_size, self.chassis_body, self.world)
        )

        # Create the person/character
        self.person = person.Person(x=x, y=y, person_width=15, person_height=30, world=self.world)

        # Create revolute joint to connect the torso body to the chassis car body
        rev_joint_def = b2RevoluteJointDef()
        joint_pos = [x / main.SCALE, y / main.SCALE]
        rev_joint_def.Initialize(bodyA=self.person.torso.body, bodyB=self.chassis_body, anchor=joint_pos)
        self.rev_joint_torso_chassis = self.world.CreateJoint(rev_joint_def)

        # Create distance joint to connect person's torso and car's chassis
        dist_joint_def = b2DistanceJointDef()
        anchor_person = [x / main.SCALE, (y - self.person.height * 2 / 3) / main.SCALE]
        anchor_car = [(x + self.chassis_width / 2) / main.SCALE, (y - self.chassis_height / 2) / main.SCALE]
        dist_joint_def.Initialize(bodyA=self.person.torso.body, bodyB=self.chassis_body, anchorA=anchor_person,
                                  anchorB=anchor_car)
        dist_joint_def.frequencyHz = 5
        dist_joint_def.dampingRatio = 0.1
        dist_joint_def.length *= 1.1
        self.dist_joint_torso_chassis = self.world.CreateJoint(dist_joint_def)

        # Set chassis_body variables
        self.chassis_body.angularDamping = 0.1
        self.chassis_body.userData = self

    # Function to set random colour of shirt
    def set_shirt_colour(self):
        self.person.torso.color = pygame.Color(r=random.randint(0, 255), g=random.randint(0, 255),
                                               b=random.randint(0, 255))

    # Function that draws/renders the person, wheels and the car on the screen
    def draw_person_car(self, screen, head_sprite, wheel_sprite, car_sprite):

        # Get position and angle of the car chassis
        x = self.chassis_body.GetPosition().x * main.SCALE
        y = self.chassis_body.GetPosition().y * main.SCALE
        angle = self.chassis_body.GetPosition().GetAngle()

        # Draw the char chassis
        car_sprite = pygame.transform.rotate(car_sprite, angle)
        screen.blit(
            source=car_sprite,
            dest=(-self.chassis_width / 2 - 7 - main.panX, self.chassis_height - 20 - main.panY,
                  self.chassis_width + 23, self.chassis_height * 2 + 10)
        )

        # Draw person on screen
        self.person.draw_person(screen, head_sprite)

        # Draw wheels on screen
        for wheel in self.wheels:
            wheel.draw_wheel(screen, wheel_sprite)

    # A function that updates whether the player status is alive or death
    def update_status(self):
        x = self.chassis_body.GetPosition().x * main.SCALE
        y = self.chassis_body.GetPosition().y * main.SCALE
        self.change_counter += 1
        # Check whether we are moving forward with the car
        if x > self.max_distance:
            self.max_distance = x
            if math.floor(self.max_distance) % 50 == 0:
                self.change_counter = 0
        else:  # When no significant distance has been made we set player status to dead
            if self.change_counter > 250:
                if not main.HUMAN_PLAYING:
                    self.player.dead = True
        # When player is out of the screen height, we set status to dead
        if not self.dead and y > main.SCREEN_HEIGHT:
            self.dead = True
            self.player.dead = True

    # Function that turns on the motor on wheels and moves forward
    def motor_on(self, forward):
        self.wheels[0].joint.EnableMotor(True)
        self.wheels[1].joint.EnableMotor(True)
        old_state = self.motor_state
        if forward:  # When we move forward / give gas
            self.motor_state = 1
            self.wheels[0].joint.SetMotorSpeed(-self.motor_speed * math.pi)
            self.wheels[1].joint.SetMotorSpeed(-self.motor_speed * math.pi)
            self.chassis_body.ApplyTorque(-self.rotation_torque)
        else:   # When not giving gas we slow down
            self.motor_state = -1
            self.wheels[0].joint.SetMotorSpeed(self.motor_speed * math.pi)
            self.wheels[1].joint.SetMotorSpeed(self.motor_speed * math.pi)
        # Rotation applied to the car when we stop giving gas
        if old_state + self.motor_state == 0:
            if old_state == 1:
                self.chassis_body.ApplyTorque(self.motor_state * -1 * self.rotation_torque)

        # Set maximum motor torque on wheels
        self.wheels[0].joint.SetMaxMotorTorque(700)
        self.wheels[1].joint.SetMaxMotorTorque(350)

    # When we brake, we turned the motor off, that will also apply torque
    def motor_off(self):
        if self.motor_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * self.rotation_torque)
        self.motor_state = 0
        self.wheels[0].joint.EnableMotor(False)
        self.wheels[1].joint.EnableMotor(False)

