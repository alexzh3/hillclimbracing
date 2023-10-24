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
        self.torso = Torso(x, y - (person_height / 2), person_width, self.world)
        self.head = Head(x, y - (person_height + person_width * 1), person_width * 1, self.world)

        rev_joint_def = b2RevoluteJointDef()
        joint_pos = b2Vec2(x / main.SCALE, (y - person_height) / main.SCALE)
        rev_joint_def.__init__(self.head.body, self.torso.body, joint_pos)
        self.head_joint = self.world.CreateJoint(rev_joint_def)
        





class Head:
    def __init__(self, x, y, r, world=None):
        ...

class Torso:
    ...
