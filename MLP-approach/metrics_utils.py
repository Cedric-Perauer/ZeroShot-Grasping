
import torch 
from shapely.geometry import Polygon
import math
def create_oriented_bounding_box(point1, point2, h):
    direction_vector = point2 - point1
    direction_vector /= torch.norm(direction_vector.to(torch.float32))

    perpendicular_vector = torch.tensor([direction_vector[1], -direction_vector[0]])
    scaled_perpendicular_vector = h * perpendicular_vector

    corner_points = torch.stack(
        [
            point1 + scaled_perpendicular_vector,
            point1 - scaled_perpendicular_vector,
            point2 - scaled_perpendicular_vector,
            point2 + scaled_perpendicular_vector,
        ]
    )

    return corner_points

def oriented_bounding_box_iou(box1, box2):
    # Convert tensors to lists
    box1 = box1.tolist()
    box2 = box2.tolist()
    # Create polygons from the corner points
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    # Calculate intersection area
    intersection = poly1.intersection(poly2).area

    # Calculate union area
    union = poly1.area + poly2.area - intersection

    # Compute IoU
    iou = intersection / (union + 1e-6)
   

    return iou


def grasp_correct_full(pred_point, single_point, gt_grasp,heights,thresh_angle=30,thresh_iou=0.25):
        '''
        given the input point single_point we check if the second point is a valid grasp 
        for that : 
        1) angle difference is less than 30 degress 
        2) the IOU of the points is at least 0.25 
        '''
        
        angle_flag = False
        iou_flag = False
        
        ##1) angle verifier 
        vec_pred_to_single = (pred_point - single_point).to(torch.float64) #vector of the prediction grasps
        vec_single_to_gt = (gt_grasp[0] - gt_grasp[1]).to(torch.float64) #vector of the ground truth grasps
        vec_single_to_gt_rev = (gt_grasp[1] - gt_grasp[0]).to(torch.float64) #vector of the ground truth grasps


        vec_pred_to_single = vec_pred_to_single/torch.norm(vec_pred_to_single)
        vec_single_to_gt = vec_single_to_gt / torch.norm(vec_single_to_gt)
        vec_single_to_gt_rev = vec_single_to_gt_rev / torch.norm(vec_single_to_gt_rev)



        dot_product = torch.dot(vec_pred_to_single, vec_single_to_gt)
        dot_product_rev = torch.dot(vec_pred_to_single, vec_single_to_gt_rev)
        # Calculate the magnitudes of the vectors
        magnitude_pred_to_single = torch.norm(vec_pred_to_single.to(torch.float64))
        magnitude_single_to_gt = torch.norm(vec_single_to_gt.to(torch.float64))
        magnitude_single_to_gt_rev = torch.norm(vec_single_to_gt_rev.to(torch.float64))

        # Calculate the cosine of the angle between the vectors
        cos_angle = dot_product / (magnitude_pred_to_single * magnitude_single_to_gt)
        cos_angle_rev = dot_product_rev / (magnitude_pred_to_single * magnitude_single_to_gt_rev)

        # Calculate the angle in radians
        correct = False
        angle_radians = torch.acos(dot_product)
        angle_radians_rev = torch.acos(dot_product_rev)

        # Convert the angle to degrees
        angle_degrees = angle_radians * (180.0 / torch.pi)
        angle_degrees_rev = angle_radians_rev * (180.0 / torch.pi)
        #print("angle_degrees", angle_degrees)
        #print("angle_degrees_rev", angle_degrees_rev)

        angle_return = min(angle_degrees, angle_degrees_rev)
        if angle_degrees < thresh_angle or angle_degrees_rev < thresh_angle:
                angle_flag = True
        
        if math.isnan(angle_degrees) or math.isnan(angle_degrees_rev):
                angle_flag = True
                angle_return = torch.tensor(0, dtype=torch.float64)

        
        corner_points_gt = create_oriented_bounding_box(gt_grasp[0].to(torch.float32),gt_grasp[1].to(torch.float32),heights)
        corner_points_pred = create_oriented_bounding_box(single_point.to(torch.float32),pred_point.to(torch.float32),heights)
        
        try : 
                iou = oriented_bounding_box_iou(corner_points_gt, corner_points_pred)
        except :
                iou = 0 
        

        if iou >= thresh_iou:
                iou_flag = True
        
        if angle_flag == True and iou_flag == True:
                correct = True
        return corner_points_gt, corner_points_pred, correct, iou, angle_return
        