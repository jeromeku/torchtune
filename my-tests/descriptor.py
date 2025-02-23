class Descriptor:
    def __init__(self, initial_value=None):
        self.name = None  # Will store the attribute name
        self.initial_value = initial_value

    def __set_name__(self, owner, name):
        # This gets called when the descriptor is assigned to a class
        print(f"Setting name {self} {owner} {name}")
        self.name = name

    def __get__(self, instance, owner):
        print(f"Getting {self} {instance} {owner}")
        if instance is None:
            # Accessed via the class: return the descriptor itself
            return self
        # Get the value from the instance's __dict__
        return instance.__dict__.get(self.name, self.initial_value)

    def __set__(self, instance, value):
        print(f"Setting {self} {instance} {value}")
        # Store the value in the instance's __dict__
        instance.__dict__[self.name] = value

class MyClass:
    # Attach the descriptor to a class attribute
    attr = Descriptor(initial_value=42)

obj = MyClass()
obj.attr = 100      # Now this will trigger __set__
print(obj.attr)     # And this will show 100