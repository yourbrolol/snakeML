from itertools import product

def indices(shape): yield from product(*(range(s) for s in shape)) if shape != () else ()
