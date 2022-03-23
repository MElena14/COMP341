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
    r1 = RotatedRect(*rect1)
    r2 = RotatedRect(*rect2)
    
    intersection = r1.intersection(r2).area
    union = (r1.width*r1.height) + (r2.width*r2.height) - intersection
    score = intersection/union
    return score

def grasp_accuracy(Ys, Y_hats):
    scores = [compare_grasps(Ys[i], Y_hats[i]) for i, g in enumerate(Ys)]
    print(scores)
    avg = lambda x: sum(x)/len(x)
    return avg(scores)


def visul_grasps(rect1, rect2):
    r1 = RotatedRect(*rect1)
    r2 = RotatedRect(*rect2)
    print("score: ", compare_grasps(rect1, rect2))
    
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


