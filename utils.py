import tensorflow as tf
from keras import backend as K
import sys


def bb_intersection_over_union(predA, predB):

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

    xA = K.maximum(boxA[0], boxB[0])
    yA = K.maximum(boxA[1], boxB[1])
    xB = K.minimum(boxA[2], boxB[2])
    yB = K.minimum(boxA[3], boxB[3])

    interArea = K.maximum(0, xB - xA + 1) * K.maximum(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)


    iou = K.cast(interArea, 'float') / K.cast(boxAArea + boxBArea - interArea, "float")
    

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
    
  
    cls_loss = K.sum(K.square(y_true_class - y_pred_class), axis= -1) * y_true_conf
    xy_loss = K.sum(K.square(y_true_xy - y_pred_xy), axis= -1) * y_true_conf
    
    wh_loss = K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis= -1) * y_true_conf

    
    iou, _ = bb_intersection_over_union(y_true, y_pred)
    conf_loss = K.square(y_true_conf*iou - y_pred_conf)* y_true_conf

    total_loss = xy_loss + conf_loss + cls_loss + wh_loss
    
    conf_val = K.argmax(K.eval(conf_loss), axis = -1)
    cls_val = K.argmax(K.eval(cls_loss), axis = -1)
    xy_val = K.argmax(K.eval(xy_loss), axis = -1)
    wh_val = K.argmax(K.eval(wh_loss), axis = -1)
    iou_val = K.argmax(K.eval(iou), axis = -1)
    y_true_val = K.argmax(K.eval(y_true_conf), axis = -1)
    total_loss_val = K.argmax(K.eval(total_loss), axis = -1) 

    tf.print(f"conf_loss: {total_loss_val} conf_loss: {conf_val}, y_true_val: {y_true_val}, xy_loss: {xy_val}, wh_loss: {wh_val}, iou: {iou_val} ", output_stream=sys.stdout)
    tf.print(f"conf_loss: {conf_loss}, xy_loss: {xy_loss}, wh_loss: {wh_loss}, iou: {iou} ", output_stream=sys.stdout)
    
    # tf.print(f"total_loss : {total_loss}", output_stream=sys.stdout)
    return total_loss