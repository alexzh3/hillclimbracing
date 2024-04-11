import car
import hill_racing


class Agent:
    def __init__(self, real_world):
        self.dead = False  # Whether the agent is dead
        self.score = 0
        self.world = real_world
        self.last_grounded = 0
        self.car = None
        self.motor_state = 2
        self.spawning_x = hill_racing.SPAWNING_X  # Spawn location
        self.spawning_y = hill_racing.SPAWNING_Y
        self.steps_in_air = 0

    def add_to_world(self):
        self.car = car.Car(x=self.spawning_x, y=self.spawning_y, world=self.world, agent=self)

    def draw_agent(self, surface_screen):
        if not self.dead:  # Draw car when agent has died less than dead count amount
            self.car.draw_person_car(surface_screen)

    def update(self):
        # print(self.car.chassis_body.position.x, self.car.max_distance)
        # print(self.car.wheels[0].joint.speed, self.car.wheels[1].joint.speed)
        # Update the panX and panY offset for camera
        hill_racing.panX = self.car.chassis_body.position.x * hill_racing.SCALE - 100
        # hill_racing.panY = self.car.chassis_body.position.y * hill_racing.SCALE - hill_racing.SPAWNING_Y

        if self.car.dead:  # If the car is dead
            self.dead = True
        elif not self.car.dead:
            self.car.update_status()

        # Calculate and update score, score is equal to at least 0 or current max distance - spawn position
        self.score = max(0, int((self.car.max_distance - (self.spawning_x / hill_racing.SCALE))))
        # Count number of steps where agent is in the air
        if not self.car.wheels[0].on_ground and not self.car.wheels[1].on_ground:
            self.steps_in_air += 1

    # Function that removes body from world
    def remove_agent_from_world(self):
        self.destroy_agent()

    def reset_car(self):
        self.destroy_agent()
        self.car = car.Car(self.spawning_x, self.spawning_y, self.world)

    def destroy_agent(self):
        self.world.DestroyBody(self.car.chassis_body)
        self.world.DestroyBody(self.car.wheels[0].body)
        self.world.DestroyBody(self.car.wheels[0].rim_body)
        self.world.DestroyBody(self.car.wheels[1].body)
        self.world.DestroyBody(self.car.wheels[1].rim_body)
        self.world.DestroyBody(self.car.person.head.body)
        self.world.DestroyBody(self.car.person.torso.body)
        # self.world.DestroyJoint(self.car.dist_joint_torso_chassis)
        # self.world.DestroyJoint(self.car.person.dist_joint_head_torso)
        # self.world.DestroyJoint(self.car.rev_joint_torso_chassis)
        # self.world.DestroyJoint(self.car.person.rev_joint_head_torso)
