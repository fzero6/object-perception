#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs:
    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # perform outlier filtering first to clean up the images
    outlier_filter = cloud.make_statistical_outlier_filter()
    # set the number of points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # set threshold scale factor; tested values: 1.0
    x = 0.1
    # any point with a mean distance large than the global (mean distance + x * std_dev)
    # will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # call the filter function
    cloud_filtered = outlier_filter.filter()
    # debug checking
    filename = './debug/outlier_removal.pcd'
    pcl.save(cloud_filtered, filename)


    # create a voxelgrid filter object for our input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.005
    # set the voxel (or leaf)size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    filename = './debug/voxel_downsampled.pcd'
    pcl.save(cloud_filtered, filename)

    # PassThrough filter
    # create pass through filter object
    passthrough = cloud_filtered.make_passthrough_filter()
    # set filter axis to z to isolate the table
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.7  # trials: 0.6
    axis_max = 1.0  # trials: 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    # generate the resultant point cloud
    cloud_filtered = passthrough.filter()
    filename = './debug/passthrough_filter_z.pcd'
    pcl.save(cloud_filtered, filename)

    # PassThrough filter
    # create pass through filter object
    passthrough = cloud_filtered.make_passthrough_filter()
    # set filter axis to y to isolate the table objects on table and remove peripheral edges of table
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4  # trials: -0.5(too large)
    axis_max = 0.4  # trials: 0.5
    passthrough.set_filter_limits(axis_min, axis_max)
    # generate the resultant point cloud; was cloud_filtered =
    cloud_objects = passthrough.filter()
    filename = './debug/passthrough_filter_y.pcd'
    pcl.save(cloud_objects, filename)
    '''
    # RANSAC plane segmentation
    # create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # set the model to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract table
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    filename = './debug/extracted_outliers.pcd'
    pcl.save(cloud_table, filename)

    # Extract outliers
    # create the filter objectd
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    filename = './debug/extracted_inliers.pcd'
    pcl.save(cloud_objects, filename)
    '''

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # create cluster extraction
    ec = white_cloud.make_EuclideanClusterExtraction()
    # set the tolerances for
    # minimum and maximum cluster sizes bound how big the cluster for each detected object will be.
    # if max value is too small it will make too many clusters
    # if the minimum value is too large it will miss clusters
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(2000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # assign colors to the individual point cloud indices for visualization
    cluster_color = get_color_list(len(cluster_indices))

    # create an empty list to store the labels and point clouds
    color_cluster_point_list = []
    detected_objects_labels = []
    detected_objects = []

    for j, indices in enumerate(cluster_indices):

        # initialize a list of
        object_clusters = []

        for i, indice in enumerate(indices):
            # extract point cloud information for each object (x, y, z, RGB)
            object_clusters.append([cloud_objects[indice][0],
                                    cloud_objects[indice][1],
                                    cloud_objects[indice][2],
                                    cloud_objects[indice][3]])
            # extract point cloud informtion and assign RGB value to points in cluster
            # the RBG value is determined using the get_color_list function which is random
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

        # convert the cluster from pcl to ROS using helper function
        pcl_cluster = pcl.PointCloud_PointXYZRGB()
        pcl_cluster.from_list(object_clusters)

        # convert pcl to ros format
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract the histogram features from the cloud
        # same way as done in capture_features.py
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=False)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        # retrieve the corresponding label and add it to detected_objects_labels
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # add detected objects to the list of detected objects
        label_pos = list(white_cloud[indices[0]])
        label_pos[2] += 0.4
        object_markers_pub.publish(make_label(label, label_pos, j))

        # add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    filename = './debug/cluster_clouds.pcd'
    pcl.save(cluster_cloud, filename)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    # ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    # pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # set the filename for the output yaml file
    filename = "output_1.yaml"

    # initializing list for detected object labels
    labels = []

    #initializing list of tuples for the centroids (x, y, z)
    centroids = []
    yaml_list = []

    #intitialize the data types for the test_scene_number and object_name variables
    test_scene_num  = Int32()
    object_name = String()


    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    # TODO: Parse parameters into individual variables
    #object_name = object_list_param[i]['name']
    #object_group = object_list_param[i]['group']
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for i in object_list_param:

        # separate the object name and group variables to iterate through
        object_name = i['name']
        object_group = i['group']

        # for loop to iterate the detected objects
        for j, detected_val in enumerate(object_list):

            if object_name != detected_val.label:
                # if the object from the pick list doesn't match any of the detected objects, skip that item.
                continue

            # TODO: Calculate the centroid of the object to pick
            # add the object list label to the labels list. object_list.label points to detected object
            labels.append(object_list.label)
            # convert the object to an array
            points_arr = ros_to_pcl(object_list.cloud).to_array()
            # compute the centroid of the object, data will be float64 => convert to python/ROS format
            centroids.append(np.asscalar(np.mean(points_arr, axis=0))[:3])

            # set the test scene number or grab it from the
            scene_number = 1
            test_scene_num.data = scene_number

            # set the object name for the current object
            object_name.data = object_list_param[i]['name']


            # TODO: Create 'place_pose' for the object
            # TODO: Assign the arm to be used for pick_place

            # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            pick_pose = Pose()
            pick_pose.position.x = centroids[0]
            pick_pose.position.y = centroids[1]
            pick_pose.position.z = centroids[2]

            # MAKE THE YAML FILE
            # input format for the yaml function below
            # make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
            yaml_dict = make_yaml_dict(None,None,None,pick_pose,None)

            yaml_list.append(yaml_dict)

    send_to_yaml(filename, yaml_list)

        # Wait for 'pick_place_routine' service to come up
        #rospy.wait_for_service('pick_place_routine')
'''
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
'''


if __name__ == '__main__':

    # TODO: ROS node initialization; Exercise 2
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    # pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    # Create publishers for object markers and detected objects
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk; Exercise 3
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()