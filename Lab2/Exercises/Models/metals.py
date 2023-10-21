class metal:
    def __init__(self, sign, durability):
        self.sign = sign
        self.durability = durability


def create_set():
    return [
        metal('x', 0.09),
        metal('y', 0.06),
        metal('z', 0.99),
        metal('u', 0.98),
        metal('v', 0.1),
        metal('w', 0.15)
    ]
