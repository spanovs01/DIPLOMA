
from openvino.runtime import Core
import torchvision.transforms as transforms
import cv2
import torchvision
import numpy as np
from ultralytics.yolo.utils.plotting import colors
import random
from ultralytics.yolo.utils import ops
import torch

print(torch.__version__)
print(cv2.__version__)
print(torchvision.__version__)

def plot_one_box(box:np.ndarray, img:np.ndarray, color = (255,0,0), mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
        img (no.ndarray): input image
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if mask is not None:
        image_with_mask = img.copy()
        mask
        cv2.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
        img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
    return img

def draw_results(results, source_image:np.ndarray, label_map):
    """
    Helper function for drawing bounding boxes on image
    Parameters:
        image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
        source_image (np.ndarray): input image for drawing
        label_map; (Dict[int, str]): label_id to class name mapping
    Returns:

    """
    results = results[0]
    boxes = results["det"]
    masks = results.get("segment")
    h, w = source_image.shape[:2]
    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
        label = f'{label_map[int(lbl)]} {conf:.2f}'
        mask = masks[idx] if masks is not None else None
        source_image = plot_one_box(xyxy, source_image, mask=mask, label=label, color=(255, 157, 151), line_thickness=1)
        print()
    return source_image


def postprocess(
    pred_boxes:np.ndarray,
    input_hw,
    orig_img:np.ndarray,
    min_conf_threshold:float = 0.25,
    nms_iou_threshold:float = 0.7,
    agnosting_nms:bool = False,
    max_detections:int = 300,
    pred_masks:np.ndarray = None,
    retina_mask:bool = False
):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
        pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
        retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    # if pred_masks is not None:
    #     nms_kwargs["nm"] = 32
    # print(f"BEFORE NMS preds are here: {pred_boxes.shape}")
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=1,
        **nms_kwargs
    )
    # print(f"AFTER NMS preds are here: {len(preds)}, {preds[0]}, {len(preds[0][0])}")
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    for i, pred in enumerate(preds):
        print(f"PRED = {pred}")
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            print(f"PROTOS[i] == {proto[i].shape}")#, {proto[i]}")
            print(f"PRED[:, 6:] == {pred[:, 6:].shape}")#, {pred[:, 6:]}")
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [ops.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results

def inference(model_xml_path, im_path):

    label_map = {0: 'red', 1 : 'green'}
    print(colors(1))
    ie = Core()
    # model_xml_path = "/home/ss21mipt/Documents/starkit/DIPLOMA/to_rhoban/weights/Feds_yolov8_2_openvino/best.xml"
    # im_path = "/home/ss21mipt/Documents/starkit/field.jpg"
    model = ie.read_model(model_xml_path)
    device = "CPU"
    compiled_model = ie.compile_model(model=model, device_name=device)
    print("INPUT TENSOR ", compiled_model.input().shape, compiled_model.output(0).shape)#, compiled_model.output(1).shape, type(compiled_model))
    cam = cv2.VideoCapture(0)
    key = cv2.waitKey(1)

    while key != 27:
        # _, image = cam.read()
        # cv2.imshow("source", image)
        image = cv2.imread(im_path)
        image = cv2.resize(image, (640,640), interpolation=cv2.INTER_LINEAR)
        im = np.expand_dims(image, axis=0)
        im_for_draw = image
        im = np.transpose(im, (0,3,1,2))
        input_tensor = im.astype(np.float32)  # uint8 to fp32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        print(f"image\t{type(im), im.shape, np.info(im)}")
        print(f"im\t{type(im), im.shape, np.info(im)}")
        print(f"input_tensor\t{type(input_tensor), input_tensor.shape, np.info(input_tensor)}")

        result = compiled_model([input_tensor])
        output_det = compiled_model.output(0)
        output_seg = compiled_model.output(1)
        boxes = compiled_model([input_tensor])[output_det]
        masks = compiled_model([input_tensor])[output_seg]  
        results = postprocess(boxes, (640,640), image, pred_masks=masks)
        proto = torch.from_numpy(masks) if masks is not None else None
        detections = postprocess(pred_boxes=boxes, input_hw=(640,640), orig_img=image, pred_masks=masks)
        res_img = draw_results(detections, im_for_draw, label_map)
        # cv2.imshow("source" ,image)
        cv2.imshow("res" , res_img)
        if cv2.waitKey(1) == 27:
            cam.release()
            cv2.destroyAllWindows()
            break
if __name__ == '__main__':
    # xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/to_rhoban/weights/Feds_yolov8_2_openvino/best.xml"
    # xml = "/home/ss21mipt/Downloads/best.xml"
    xml = "/home/ss21mipt/DIPLOMA/weights/best_openvino_model/best.xml"
    png = "/home/ss21mipt/Pictures/photo_2023-03-28_12-46-25.jpg"
    inference(xml, png)
# print()
# input_image = transforms.Compose([
#         DEFAULT_TRANSFORMS,
#         Resize(network_image_height)])((
#             image,
#             np.empty((1, 5)),
#             np.empty((network_image_height, network_image_height), dtype=np.uint8)))[0].unsqueeze(0)

# res = model("/home/ss21mipt/Documents/starkit/field", retina_masks=True)
# print(type(res))