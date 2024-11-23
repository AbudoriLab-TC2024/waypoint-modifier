import math

r = 10.0
interval = 0.2
num_points = int(2 * math.pi * r / interval)
print("x,y,action")
points = []
for i in range(num_points):
    x = r * math.cos(i * interval / r - math.pi / 2)
    y = r * math.sin(i * interval / r - math.pi / 2) + r
    print(f"{x:.3f},{y:.3f}")

print("0.000,0.000")
