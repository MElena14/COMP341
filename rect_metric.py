import shapely.geometry
import shapely.affinity
import torch

class RotatedRect:
    def __init__(self, centerx, centery, angle, width, height):
        self.centerx = centerx
        self.centery = centery
        self.angle = angle
        self.width = abs(width)
        self.height = abs(height) #make height absolute

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
    return abs(score)

def find_max_accuracy(Y, allGrasps):
    scores = torch.zeros(len(allGrasps))
    for i, grasp in enumerate(allGrasps):
        scores[i] = compare_grasps(Y, grasp)
    return torch.max(scores)

def grasp_accuracy(Ys, allGrasps, average=True):
    scores = [find_max_accuracy(Y, allGrasps[i]) for i, Y in enumerate(Ys)]
    if average:
        avg = lambda x: sum(x)/len(x)
        return avg(scores)
    else:
        return scores


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

# find the max grasp hight and width 
# def maxHW():
#     maxH = 0
#     maxW = 0
#     dataset = get_train_loader(4)
#     for image, _, label, allGrasps in dataset:
#         for graspBatch in allGrasps:
#             for grasp in graspBatch:
#                 x, y, angle, width, height = grasp
#                 if width > maxW:
#                     maxW = width
#                 if height > maxH:
#                     maxH = height
#    return maxH, maxW
# out = (93.6522, 271.5000)

# def graspScale(graspTensorBatch):
#     newGraspTensorBatch = torch.zeros(graspTensorBatch.size())
#     for i, graspTensor in enumerate(graspTensorBatch):
#         x, y, angle, width, height = graspTensor
#         x = x/1024
#         y = y/1024
#         angle = (angle / 180) + 0.5 # angle between +90 -> -90
#         width = width / 271.5000
#         height = height / 93.6522
#         newGraspTensorBatch[i] = torch.Tensor([x, y, angle, width, height])
#     return newGraspTensorBatch

# def graspUnScale(graspTensorBatch):
#     newGraspTensorBatch = torch.zeros(graspTensorBatch.size())
#     for i, graspTensor in enumerate(graspTensorBatch):
#         x, y, angle, width, height = graspTensor
#         newGraspTensorBatch[i][0] = x*1024
#         newGraspTensorBatch[i][1] = y*1024
#         newGraspTensorBatch[i][2] = (angle - 0.5) * 180 # angle between +90 -> -90
#         newGraspTensorBatch[i][3] = width * 271.5000
#         newGraspTensorBatch[i][4] = height * 93.6522
#     return newGraspTensorBatch