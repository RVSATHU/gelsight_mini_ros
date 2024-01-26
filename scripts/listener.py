#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import rospy
import matplotlib.pyplot as plt
# import msg file
from gelsight_mini_ros.msg import tracking_msg, judging_msg

left_marker_init_x = None
left_marker_init_y = None
left_marker_x = None
left_marker_y = None
right_marker_init_x = None
right_marker_init_y = None
right_marker_x = None
right_marker_y = None


def callback_left(data):
    global left_marker_x, left_marker_y, left_marker_init_x, left_marker_init_y
    left_marker_init_x = np.array(data.marker_x)
    left_marker_init_y = np.array(data.marker_y)
    left_marker_x = np.array(data.marker_x) + np.array(data.marker_displacement_x)
    left_marker_y = np.array(data.marker_y) + np.array(data.marker_displacement_y)


def callback_right(data):
    global right_marker_x, right_marker_y, right_marker_init_x, right_marker_init_y
    right_marker_init_x = np.array(data.marker_x)
    right_marker_init_y = np.array(data.marker_y)
    right_marker_x = np.array(data.marker_x) + np.array(data.marker_displacement_x)
    right_marker_y = np.array(data.marker_y) + np.array(data.marker_displacement_y)


def callback_jdg(data):
    # rospy.loginfo('Marker_Tracking_Srv Listened!')
    if data.is_contact:
        print("contact")


def listener():
    rospy.init_node('gelLis')
    rospy.Subscriber('Marker_Tracking_Left', tracking_msg, callback_left)
    rospy.Subscriber('Marker_Tracking_Right', tracking_msg, callback_right)
    rospy.Subscriber('Marker_Tracking_Contact', judging_msg, callback_jdg)
    # boost_clt.init()
    rospy.logwarn('Start listening!')
    plt.figure()
    plt.ion()
    while not rospy.is_shutdown():
        plt.clf()
        ax1 = plt.subplot(121)
        plt.scatter(left_marker_init_x, left_marker_init_y)
        plt.scatter(left_marker_x, left_marker_y)
        ax1.invert_yaxis()

        ax2 = plt.subplot(122)
        plt.scatter(right_marker_init_x, right_marker_init_y)
        plt.scatter(right_marker_x, right_marker_y)
        ax2.invert_yaxis()
        plt.pause(0.01)
        # rospy.spin()


if __name__ == '__main__':
    listener()
