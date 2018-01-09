#!/usr/bin/env python

# Import modules
from pcl_helper import *

# TODO: Define functions as required
from pcl_helper import ros_to_pcl
from pcl_helper import pcl_to_ros
from pcl_helper import get_color_list
from pcl_helper import rgb_to_float

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

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
	#filename = 'voxel_downsampled.pcd'
	#pcl.save(cloud_filtered, filename)

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
	#filename = 'cloud_table.pcd'
	#pcl.save(cloud_table, filename)

	# Extract objects
	#cloud_objects = cloud_filtered.extract(inliers, negative=True)
	
	# Extract outliers
	# create the filter objectd
	objects = cloud_filtered.extract(inliers, negative = True)
		
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
	
	filename = 'cloud_objects.pcd'
	pcl.save(cloud_objects, filename)
	
    # TODO: Euclidean Clustering 
	white_cloud = XYZRGB_to_XYZ(cloud_objects)
	tree = white_cloud.make_kdtree()

	# create cluster extraction
	ec = white_cloud.make_EuclideanClusterExtraction()

	# set the tolerances for
	ec.set_ClusterTolerance(0.05)
	ec.set_MinClusterSize(100)
	ec.set_MaxClusterSize(2000)
	ec.set_SearchMethod(tree)
	cluster_indices = ec.Extract()
	
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    
	# asign colors to the individual point cloud indices for visualization
	cluster_color = get_color_list(len(cluster_indices))

	color_cluster_point_list = []

	for j, indices in enumerate(cluster_indices):
		for i, indice in enumerate(indices):
			color_cluster_point_list.append([white_cloud[indice][0],
											white_cloud[indice][1],
											white_cloud[indice][2],
											rgb_to_float(cluster_color[j])])

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

if __name__ == '__main__':

    # TODO: ROS node initialization
	rospy.init_node('clustering', anonymous=True)
    # TODO: Create Subscribers
	pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)
    # TODO: Create Publishers
	pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
	pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
	pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    # Initialize color_list
	get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
	while not rospy.is_shutdown():
 		rospy.spin()
