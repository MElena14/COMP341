import shapely.geometry
import shapely.affinity

class RotatedRect:
    def __init__(self, centerx, centery, width, height, angle):
        self.centerx = centerx
        self.centery = centery
        self.width = width
        self.height = height
        self.angle = angle

    def get_contour(self):
        width = self.width
        height = self.height
        contour = shapely.geometry.box(-width/2.0, -height/2.0, width/2.0, height/2.0)
        rotContour = shapely.affinity.rotate(contour, self.angle)
        return shapely.affinity.translate(rotContour, self.centerx, self.centery)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())


r1 = RotatedRect(18, 15, 13, 20, 70)
r2 = RotatedRect(12, 10, 5, 20, 130)

intersection = r1.intersection(r2).area
union = (r1.width*r1.height) + (r2.width*r2.height) - intersection
score = intersection/union

print(intersection)
print(union)
print(score)

from matplotlib import pyplot
from descartes import PolygonPatch

fig = pyplot.figure(1, figsize=(10, 4))
ax = fig.add_subplot(121)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)

ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))

pyplot.show()