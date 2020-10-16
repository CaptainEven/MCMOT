import numpy as np
import cv2
from lib.tracker.multitracker import id2cls

# cls_color_dict = {
#     'car': [180, 105, 255],  # hot pink
#     'bicycle': [219, 112, 147],  # MediumPurple
#     'person': [98, 130, 238],  # Salmon
#     'cyclist': [181, 228, 255],
#     'tricycle': [211, 85, 186]
# }


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_detects(image,
                 dets_dict,
                 num_classes,
                 frame_id,
                 fps=0.0):
    """
    plot detection results of this frame(or image)
    :param image:
    :param dets_dict:
    :param num_classes:
    :param frame_id:
    :param fps:
    :return:
    """
    img = np.ascontiguousarray(np.copy(image))
    # im_h, im_w = img.shape[:2]

    text_scale = max(1.0, image.shape[1] / 1200.)  # 1600.
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 600.))

    for cls_id in range(num_classes):
        # plot each object class
        cls_dets = dets_dict[cls_id]

        cv2.putText(img, 'frame: %d fps: %.2f'
                    % (frame_id, fps),
                    (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 0, 255),
                    thickness=2)

        # plot each object of the object class
        for obj_i, obj in enumerate(cls_dets):
            # left, top, right, down, score, cls_id
            x1, y1, x2, y2, score, cls_id = obj
            cls_name = id2cls[int(cls_id)]
            box_int = tuple(map(int, (x1, y1, x2, y2)))
            # cls_color = cls_color_dict[cls_name]
            cls_color = get_color(abs(cls_id))

            # draw bbox for each object
            cv2.rectangle(img,
                          box_int[0:2],
                          box_int[2:4],
                          color=cls_color,
                          thickness=line_thickness)

            # draw class name
            cv2.putText(img,
                        cls_name,
                        (box_int[0], box_int[1]),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        [0, 255, 255],  # cls_id: yellow
                        thickness=text_thickness)

    return img


def plot_tracks(image,
                tlwhs_dict,
                obj_ids_dict,
                num_classes,
                scores=None,
                frame_id=0,
                fps=0.0):
    """
    :rtype:
    :param image:
    :param tlwhs_dict:
    :param obj_ids_dict:
    :param num_classes:
    :param scores:
    :param frame_id:
    :param fps:
    :return:
    """
    img = np.ascontiguousarray(np.copy(image))
    im_h, im_w = img.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1.0, image.shape[1] / 1200.)  # 1600.
    # text_thickness = 1 if text_scale > 1.1 else 1
    text_thickness = 2  # 自定义ID文本线宽
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w / 140.))

    for cls_id in range(num_classes):
        cls_tlwhs = tlwhs_dict[cls_id]
        obj_ids = obj_ids_dict[cls_id]

        cv2.putText(img, 'frame: %d fps: %.2f'
                    % (frame_id, fps),
                    (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 0, 255),
                    thickness=2)

        for i, tlwh_i in enumerate(cls_tlwhs):
            x1, y1, w, h = tlwh_i
            int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))  # x1, y1, x2, y2
            obj_id = int(obj_ids[i])
            id_text = '{}'.format(int(obj_id))

            _line_thickness = 1 if obj_id <= 0 else line_thickness
            color = get_color(abs(obj_id))
            # cls_color = cls_color_dict[id2cls[cls_id]]

            # draw bbox
            cv2.rectangle(img=img,
                          pt1=int_box[0:2],  # (x1, y1)
                          pt2=int_box[2:4],  # (x2, y2)
                          color=color,
                          thickness=line_thickness)

            # draw class name and index
            cv2.putText(img,
                        id2cls[cls_id],
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        (0, 255, 255),  # cls_id: yellow
                        thickness=text_thickness)

            txt_w, txt_h = cv2.getTextSize(id2cls[cls_id],
                                           fontFace=cv2.FONT_HERSHEY_PLAIN,
                                           fontScale=text_scale, thickness=text_thickness)

            cv2.putText(img,
                        id_text,
                        (int(x1), int(y1) - txt_h),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        (0, 255, 255),  # cls_id: yellow
                        thickness=text_thickness)

    return img


def plot_tracking(image,
                  tlwhs,
                  obj_ids,
                  scores=None,
                  frame_id=0,
                  fps=0.,
                  ids2=None,
                  cls_id=0):
    """
    :param image:
    :param tlwhs:
    :param obj_ids:
    :param scores:
    :param frame_id:
    :param fps:
    :param ids2:
    :param cls_id:
    :return:
    """
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1.0, image.shape[1] / 1200.)  # 1600.
    # text_thickness = 1 if text_scale > 1.1 else 1
    text_thickness = 2  # 自定义ID文本线宽
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d'
                % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 0, 255),
                thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))

        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))

        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, int_box[0:2], int_box[2:4], color=color, thickness=line_thickness)  # bbox: 随机颜色

        # 绘制id编号
        cv2.putText(im,
                    id_text,
                    (int_box[0], int_box[1] + 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 255, 255),  # id: yellow
                    thickness=text_thickness)

        # 绘制目标类别
        cv2.putText(im,
                    id2cls[cls_id],
                    (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 255, 255),  # cls_id: yellow
                    thickness=text_thickness)

    return im


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    """
    :param image:
    :param tlbrs:
    :param scores:
    :param color:
    :param ids:
    :return:
    """
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im
