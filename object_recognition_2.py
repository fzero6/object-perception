#!/usr/bin/env python

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


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs:
    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # create a voxelgrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    LEAF_SIZE = 0.01

    # set the voxel (or leaf)size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    # filename = 'voxel_downsampled.pcd'
    # pcl.save(cloud_filtered, filename)

    # PassThrough filter
    # create pass through filter object
    passthrough = cloud_filtered.make_passthrough_filter()

    # assign axis and ranve to the passthrough filter object
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)

    # generate the resultant point cloud
    cloud_filtered = passthrough.filter()

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
    # filename = 'cloud_table.pcd'
    # pcl.save(cloud_table, filename)

    # Extract objects
    # cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Extract outliers
    # create the filter objectd
    objects = cloud_filtered.extract(inliers, negative=True)

    outlier_filter = objects.make_statistical_outlier_filter()

    # set the number of points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # set threshold scale factor
    x = 1.0
    # any point with a mena distance large than the global (mean distance + x * std_dev)
    # will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # call the filter function
    cloud_objects = outlier_filter.filter()

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # create cluster extraction
    ec = white_cloud.make_EuclideanClusterExtraction()

    # set the tolerances for
    # minimum and maxium cluster sizes bound how big the cluster for each detected object will be.
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
        
        #initialize a list of 
        object_clusters = []
        
        for i, indice in enumerate(indices):
            #extract point cloud information for each object (x, y, z, RGB)
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
        
        #convert pcl to ros format
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

        # add the detected object to the lsit of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

    # TODO: ROS node initialization; Exercise 2
    rospy.init_node('clustering', anonymous=True)
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)
    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    # Create publishers for object markers and detected objects
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    # Initialize color_list
    get_color_list.color_list = []

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
