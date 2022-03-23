import shapely.geometry
import shapely.affinity

class RotatedRect:
    def __init__(self, centerx, centery, angle, width, height):
        self.centerx = centerx
        self.centery = centery
        self.angle = angle
        self.width = width
        self.height = height

    def get_contour(self):
        width = self.width
        height = self.height
        contour = shapely.geometry.box(-width/2.0, -height/2.0, width/2.0, height/2.0)
        rotContour = shapely.affinity.rotate(contour, self.angle)
        return shapely.affinity.translate(rotContour, self.centerx, self.centery)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

def compare_grasps(rect1, rect2):
    cx1 = rect1[0]
    cy1 = rect1[1]
    angle1 = rect1[2]
    w1 = rect1[3]
    h1 = rect1[4]
    
    cx2 = rect2[0]
    cy2 = rect2[1]
    angle2 = rect2[2]
    w2 = rect2[3]
    h2 = rect2[4]
    
    r1 = RotatedRect(cx1, cy1, angle1, w1, h1)
    r2 = RotatedRect(cx2, cy2, angle2, w2, h2)
    
    intersection = r1.intersection(r2).area
    union = (r1.width*r1.height) + (r2.width*r2.height) - intersection
    score = intersection/union
    
    print("intersection area: ", intersection)
    print("union area: ", union)
    print("score: ", score)
    
    from matplotlib import pyplot
    from descartes import PolygonPatch
    
    fig = pyplot.figure(1, figsize=(10, 4))
    ax = fig.add_subplot(121)
    ax.set_xlim(500, 800)
    ax.set_ylim(400, 700)
    
    ax.add_patch(PolygonPatch(r1.get_contour(), fc='#990000', alpha=0.7))
    ax.add_patch(PolygonPatch(r2.get_contour(), fc='#000099', alpha=0.7))
    ax.add_patch(PolygonPatch(r1.intersection(r2), fc='#009900', alpha=1))
    
    pyplot.show()

#compare_grasps([605.17693, 550.6464, 0, 186.0, 77.1998], [599.86786, 541.52308, 7.896,170.0, 95.7251])
