import tensorflow as tf
from keras import backend as K
import sys

def bb_intersection_over_union(predA, predB):
    # determine the (x, y)-coordinates of the intersection rectangle
    print("*"*40)
#     print(predA)
#     print(predB)
    b1 = predA[..., 1: 5]
    b2 = predB[..., 1: 5]
    
    
    
    x1, y1, w1, h1 = K.cast((b1[..., 0] - (b1[..., 2] / 2)) * 416, "int32"), \
                     K.cast((b1[..., 1] - (b1[..., 3] / 2)) * 416, "int32"), \
                     K.cast(b1[..., 2] * 416, "int32"), \
                     K.cast(b1[..., 3] * 416, "int32")

    x2, y2, w2, h2 = K.cast((b2[..., 0] - (b2[..., 2] / 2)) * 416, "int32"), \
                     K.cast((b2[..., 1] - (b2[..., 3] / 2)) * 416, "int32"), \
                     K.cast(b2[..., 2] * 416, "int32"), \
                     K.cast(b2[..., 3] * 416, "int32")
    
   
    boxA, boxB = [x1, y1, w1 + x1, h1 + y1], [x2, y2, w2 + x2, h2 + y2]
#     print(boxA)
#     print(boxB)
    xA = K.maximum(boxA[0], boxB[0])
    yA = K.maximum(boxA[1], boxB[1])
    xB = K.minimum(boxA[2], boxB[2])
    yB = K.minimum(boxA[3], boxB[3])
#     print([xA, yA, xB, yB])
#     if xB < xA or yB < yA:
#         return 0.0, [0, 0, 0, 0]
    # compute the area of intersection rectangle
    interArea = K.maximum(0, xB - xA + 1) * K.maximum(0, yB - yA + 1)
#     print(max(0, xB - xA + 1))
#     print(max(0, yB - yA + 1))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = K.cast(interArea, 'float') / K.cast(boxAArea + boxBArea - interArea, "float")
    
#     assert iou >= 0.0
#     assert iou <= 1.0
    # return the intersection over union value
    return iou, [xA, yA, xB, yB]

def custom_loss(y_true, y_pred):
    
    y_true_class = y_true[..., 5 : 7]
    y_pred_class = y_pred[..., 5 : 7]
    
    
    y_true_xy = y_true[..., 1 : 3]
    y_pred_xy = y_pred[..., 1 : 3]
    
    
    
    y_true_wh = y_true[..., 3 : 5]
    y_pred_wh = y_pred[..., 3 : 5]
    
    
    y_true_conf = y_true[..., 0]
    y_pred_conf = y_pred[..., 0]
    
  
    cls_loss = K.sum(K.square(y_true_class - y_pred_class), axis= -1)
    xy_loss = K.sum(K.square(y_true_xy - y_pred_xy), axis= -1) * y_true_conf
    
    wh_loss = K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis= -1) * y_true_conf

    
    iou, _ = bb_intersection_over_union(y_true, y_pred)
    conf_loss = K.square(y_true_conf*iou - y_pred_conf)

    total_loss = xy_loss + conf_loss + cls_loss + wh_loss
    
    tf.print(f"total loss: {K.sum(total_loss, axis=-1)}", output_stream=sys.stdout)
    
    return total_loss