"""
Task 1: Advanced Data Structures for a Route Planning Application
"""
from dataclasses import dataclass
import random

@dataclass
class City:
    name: str
    lat: float
    lon: float
    population: int
    distance: float

    def __repr__(self):
        return (
            f"City({self.name}, "
            f"dist={self.distance:.2f}, "
            f"pop={self.population})"
        )
    
def generate_cities(n, seed=42):
    rng = random.Random(seed)

    cities = []

    for i in range(n):

        lat = rng.uniform(-90, 90)

        lon = rng.uniform(-180, 180)

        pop = rng.randint(1000, 5000000)

        distance = rng.uniform(0, 20000)

        cities.append(
            City(
                name=f"City_{i}",
                lat=lat,
                lon=lon,
                population=pop,
                distance=distance
            )
        )

    return cities

def main():
    print("Task 1 - Advanced Data Structures")
    print("Route Planning Application")
    cities = generate_cities(5)

    for city in cities:
        print(city)


if __name__ == "__main__":
    main()