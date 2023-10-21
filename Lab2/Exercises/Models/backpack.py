class item:
    def __init__(self, name, value, weight):
        self.name = name
        self.value = value
        self.weight = weight


def create_backpack():
    return [
        item('zegar', 100, 7),
        item('obraz-pejzaz', 300, 7),
        item('obraz-portret', 200, 6),
        item('radio', 40, 2),
        item('laptop', 500, 5),
        item('lampka nocna', 70, 6),
        item('srebrne sztucce', 100, 1),
        item('porcelana', 250, 3),
        item('figura z brazu', 300, 10),
        item('skorzana torebka', 280, 3),
        item('odkurzacz', 300, 15)
    ]
