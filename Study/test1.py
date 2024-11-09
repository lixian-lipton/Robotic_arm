# -*- coding: utf-8 -*-
class Car:
    def __init__(self, make, model, year):
        self.make=make
        self.model=model
        self.year=year
    def display_details(self):
        print(f"make:{self.make},modle:{self.model},year:{self.year}")
        
#my_car=Car("bai","liu",6058)
#my_car.display_details()

class ElectricCar(Car):
    def __init__(self,make,model,year,battery_size):
        super().__init__(make,model,year)
        self.battery_size=battery_size
    def display_details(self):
        super().display_details()
        print(f"battery size:{self.battery_size}")
        
#my_electric_car=ElectricCar("An","Zhe",2658,100)
#my_electric_car.display_details()

    def _calculate_battery_life(self, distance):
        battery_life = self.battery_size * (1 - distance / 10000)#100公里消耗1%电量
        return battery_life

    def get_battery_life(self, distance):
        return self._calculate_battery_life(distance)

my_electric_car=ElectricCar("qi","yan",1037,100)
my_electric_car.display_details()

distance=5000
battery_life=my_electric_car.get_battery_life(distance)
print(f"battery life:{battery_life}")
        