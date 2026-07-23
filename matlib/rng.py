import math
import random

def normal(mean=0.0, std=1.0):
    u1 = random.random()
    u2 = random.random()

    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return mean + std * z
