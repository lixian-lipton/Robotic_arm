# -*- coding: utf-8 -*-
class Point:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __repr__(self):
        return f"Point({self.x},{self.y})"
    
Point1=Point(1,1)
Point2=Point(2,2)
Point3=Point(3,3)

point_list=[Point1,Point2,Point3]
print(f"used point_list:\n{point_list}\n")

point_list[1],point_list[2]=point_list[2],point_list[1]#第二个点与第三个点交换位置
print(f"now point_list:\n{point_list}\n")

import math

class Circle:
    def __init__(self,radius,center):
        self.radius=radius
        self.center=center
    def __repr__(self):
        return f"Circle({self.radius},{self.center}"
    
    def distance(self, other_circle):
        return math.sqrt(2*(self.center.x - other_circle.center.x) ** 2)

    def relation(self, other_circle):
        distance = self.distance(other_circle)
        r1, r2 = self.radius, other_circle.radius
        
        if distance == 0 and r1 == r2:
            print("重合")
        elif distance == r1 + r2:
            print("外切")
        elif abs(r1 - r2) < distance < r1 + r2:
            print("相交")
        elif distance == abs(r1 - r2):
            print("内切")
        elif r1 > r2 and distance < r1 - r2 or r2 > r1 and distance < r2 - r1:
            print("包含")
        else:
            print("分离")
        
Circle1=Circle(1,Point1)
Circle2=Circle(3,Point2)
Circle3=Circle(5,Point3)

circle_list=[Circle1,Circle2,Circle3]
print(f"circle_list:\n{circle_list}\n")

for i in range(len(circle_list)):
    for j in range(i+1,len(circle_list)):
        print(f"relation bewteen {circle_list[i]} and {circle_list[j]}:")
        circle_list[i].relation(circle_list[j])
        