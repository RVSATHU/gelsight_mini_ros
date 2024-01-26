#!/usr/bin/env python3
import numpy as np
import rospy
import torch
from gelsight_mini import GelSightMini
from gs_utils import demark, poisson_dct_neumaan
from cv_bridge import CvBridge

from gelsight_mini_ros.msg import judging_msg, tracking_msg  # import message file
from gelsight_mini_ros.srv import ResetMarkerTracker, ResetMarkerTrackerResponse
from sensor_msgs.msg import Image


bridge = CvBridge()

def init_msg():
    # init node
    rospy.init_node("Marker_Tracking")

    # init service
    srv_l = rospy.Service("Marker_Tracking_Srv_Left", ResetMarkerTracker, srv_callback_l)
    srv_r = rospy.Service("Marker_Tracking_Srv_Right", ResetMarkerTracker, srv_callback_r)

    # init pub
    publeft = rospy.Publisher("Marker_Tracking_Left", tracking_msg, queue_size=50)  # 创建发布者,名字
    pubright = rospy.Publisher("Marker_Tracking_Right", tracking_msg, queue_size=50)
    pubjdg = rospy.Publisher("Marker_Tracking_Contact", judging_msg, queue_size=50)
    pubimageleft = rospy.Publisher("Tactile_Image_Left", Image, queue_size=50)  # 创建发布者,名字
    pubimageright = rospy.Publisher("Tactile_Image_Right", Image, queue_size=50)
    return publeft, pubright, pubjdg, srv_l, srv_r, pubimageleft, pubimageright


# cal depth-map
def compute_depth_map(img, markermask, net, device):
    ''' Set the contact mask to everything but the markers '''

    # m.init(mc)
    cm1 = ~markermask
    '''intersection of cm and markermask '''
    ''' find contact region '''
    # cm, cmindx = np.ones(img.shape[:2]), np.where(np.ones(img.shape[:2]))
    # cmandmm = (np.logical_and(cm, markermask)).astype('uint8')

    ''' Get depth image with NN '''
    nx = np.zeros(img.shape[:2])
    ny = np.zeros(img.shape[:2])
    dm = np.zeros(img.shape[:2])

    ''' ENTIRE CONTACT MASK THRU NN '''
    # if np.where(cm)[0].shape[0] != 0:
    rgb = img[np.where(cm1)] / 255
    pxpos = np.vstack(np.where(cm1)).T
    pxpos[:, 0], pxpos[:, 1] = pxpos[:, 0] / img.shape[0], pxpos[:, 1] / img.shape[1]
    features = np.column_stack((rgb, pxpos))
    features = torch.from_numpy(features).float().to(device)

    with torch.no_grad():
        net.eval()
        out = net(features)

    nx[np.where(cm1)] = out[:, 0].cpu().detach().numpy()
    ny[np.where(cm1)] = out[:, 1].cpu().detach().numpy()

    '''OPTION#2 calculate gx, gy from nx, ny. '''
    nz = np.sqrt(1 - nx ** 2 - ny ** 2)
    if np.isnan(nz).any():
        print('nan found')

    nz = np.nan_to_num(nz)
    gx = -np.divide(nx, nz)
    gy = -np.divide(ny, nz)

    #   dilated_mm = dilate(markermask, ksize=3, iter=2)
    gx_interp, gy_interp = demark(gx, gy, markermask)

    dm1 = poisson_dct_neumaan(gx_interp, gy_interp)
    dm1 = np.array(dm1)
    # print(dm1.shape)
    return dm1


global reset_marker_tracker_left, reset_marker_tracker_right
reset_marker_tracker_right = False
reset_marker_tracker_left = False


def srv_callback_l(request):
    global reset_marker_tracker_left
    reset_marker_tracker_left = True
    return ResetMarkerTrackerResponse(True)


def srv_callback_r(request):
    global reset_marker_tracker_right
    reset_marker_tracker_right = True
    return ResetMarkerTrackerResponse(True)


def contact_jdg(x_left, y_left, x_right, y_right, dx1, dy1, dx2, dy2):
    msg_jdg = judging_msg()
    msg_jdg.is_contact = False
    msg_jdg.is_overforced = False

    dx1 = dx1 - x_left
    dy1 = dy1 - y_left
    dx2 = dx2 - x_right
    dy2 = dy2 - y_right

    dm1 = np.multiply(dx1, dx1) + np.multiply(dy1, dy1)
    dm2 = np.multiply(dx2, dx2) + np.multiply(dy2, dy2)
    dm1 = np.sqrt(dm1)
    dm2 = np.sqrt(dm2)

    if np.max(dm1) < 0.5:
        if np.max(dm2) < 0.5:
            msg_jdg.is_contact = False
        else:
            msg_jdg.is_contact = True
    else:
        msg_jdg.is_contact = True

    if np.max(dm1) < 30:
        if np.max(dm2) < 30:
            msg_jdg.is_overforced = False
        else:
            msg_jdg.is_overforced = True
    else:
        msg_jdg.is_overforced = True

    # print(np.max(dm1))

    return msg_jdg


def contact_jdg2(left_dict, right_dict, is_contact_threshold, is_overforced_threhshold):
    msg_jdg = judging_msg()
    displacement_norm_left = np.linalg.norm(left_dict["curr_marker"] - left_dict["init_marker"], 2, axis=1)
    displacement_norm_right = np.linalg.norm(right_dict["curr_marker"] - right_dict["init_marker"], 2, axis=1)

    d_left = displacement_norm_left.max()
    d_right = displacement_norm_right.max()

    msg_jdg.is_contact = d_left > is_contact_threshold or d_right > is_contact_threshold
    msg_jdg.is_overforced = d_left > is_overforced_threhshold or d_right > is_overforced_threhshold

    return msg_jdg


# Left cam serial num:28F0-GX7F
# Right cam serial num:28W2-TRN5

def convert_to_msg(infer_dict):
    tracking_msg_1 = tracking_msg()
    # import pdb
    # pdb.set_trace()
    tracking_msg_1.marker_x = list(map(float, infer_dict['init_marker'][:, 0]))
    tracking_msg_1.marker_y = list(map(float, infer_dict['init_marker'][:, 1]))
    tracking_msg_1.marker_displacement_x = list(
        map(float, infer_dict['curr_marker'][:, 0] - infer_dict['init_marker'][:, 0]))
    tracking_msg_1.marker_displacement_y = list(
        map(float, infer_dict['curr_marker'][:, 1] - infer_dict['init_marker'][:, 1]))
    return tracking_msg_1


def main():
    # init msg nodes
    global reset_marker_tracker_left, reset_marker_tracker_right
    pub_left, pub_right, pub_jdg, srv_l, srv_r, pub_image_left, pub_image_right = init_msg()

    gelsight_mini_left = GelSightMini("28F0-GX7F")
    gelsight_mini_right = GelSightMini("28W2-TRN5")

    # setup msg

    try:
        while not rospy.is_shutdown():
            if reset_marker_tracker_left:
                print('reset left')
                left_dict = gelsight_mini_left.inference(True)
                reset_marker_tracker_left = False
            else:
                left_dict = gelsight_mini_left.inference(False)

            if reset_marker_tracker_right:
                print('reset right')
                right_dict = gelsight_mini_right.inference(True)
                reset_marker_tracker_right = False
            else:
                right_dict = gelsight_mini_right.inference(False)

            left_msg = convert_to_msg(left_dict)
            right_msg = convert_to_msg(right_dict)

            left_image_msg = bridge.cv2_to_imgmsg(left_dict['current_frame'], "bgr8")
            right_image_msg = bridge.cv2_to_imgmsg(right_dict['current_frame'], "bgr8")

            # judge contact

            msg_jdg = judging_msg()
            msg_jdg.is_contact = left_dict["contact_depth"] or right_dict["contact_depth"]
            msg_jdg.is_overforced = left_dict["contact_movement"] or right_dict["contact_movement"]
            # if msg_jdg.is_contact:
            #     print('is_contact', left_dict["contact_depth"], right_dict["contact_depth"])
            if msg_jdg.is_overforced:
                print('is overforced', left_dict["contact_movement"], right_dict["contact_movement"])

            # publish message
            pub_left.publish(left_msg)
            pub_right.publish(right_msg)
            pub_jdg.publish(msg_jdg)
            pub_image_left.publish(left_image_msg)
            pub_image_right.publish(right_image_msg)

    except KeyboardInterrupt:
        print('Interrupted!')
        gelsight_mini_left.close()
        gelsight_mini_right.close()


if __name__ == "__main__":
    main()

