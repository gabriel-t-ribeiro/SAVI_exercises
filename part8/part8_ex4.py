#! /usr/bin/env python3

import numpy as np
import open3d as o3d


viewpoint = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" :
	[
		{
			"boundingbox_max" : [ 3.6699128165381372, 1.4274178742057815, 3.7624037267893198 ],
			"boundingbox_min" : [ -1.0396580696105957, -1.2458267211914062, -1.3300217410448241 ],
			"field_of_view" : 60.0,
			"front" : [ -0.071240193011207167, -0.50352019058478836, -0.86104137680670401 ],
			"lookat" : [ 1.3151273734637707, 0.090795576507187614, 1.2161909928722479 ],
			"up" : [ -0.11435743288880812, -0.85341957792307033, 0.50852473053008096 ],
			"zoom" : 0.57000000000000006
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def build_table_frame(table_normal: np.ndarray, table_center: np.ndarray) -> np.ndarray:
    """Build 4x4 T_scene_table from an upward unit normal and tabletop centroid."""
    # TODO (Step 3): Construct orthonormal basis (x_axis, y_axis, z_axis) and 4x4 matrix
    # Ensure z_axis is a unit vector
    z_axis = table_normal / np.linalg.norm(table_normal)

    # Pick a non-collinear helper vector
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

    # Construct orthonormal right-handed basis (x, y, z)
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Assemble 4x4 matrix T_scene_table
    T_scene_table = np.eye(4)
    T_scene_table[:3, 0] = x_axis
    T_scene_table[:3, 1] = y_axis
    T_scene_table[:3, 2] = z_axis
    T_scene_table[:3, 3] = table_center
    return T_scene_table


def main():
    # Load full scene point cloud
    pcd = o3d.io.read_point_cloud("scene_point_cloud/scene.ply")
    print("Loaded scene:", pcd)

    # --- Step 1: Ground Plane Detection & Normal Sign Check ---
    # Run scene.segment_plane(...) to get floor_model [a, b, c, d] and floor_inliers
    plane, inliers = pcd.segment_plane(
        distance_threshold=0.05,  # Thickness tolerance (meters) around the fitted plane
        ransac_n=3,  # Number of points to sample per trial (keep 3 for planes)
        num_iterations=1000  # Number of random trials
    )

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # red

    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # Check signed heights h(p) = n . p + d on non-floor points; negate floor_model if mean < 0
    pts = np.asarray(outlier_cloud.points)
    heights = pts @ plane[:3] + plane[3]
    if np.mean(heights) < 0:
        plane = -plane  # negate ALL 4 coefficients [a, b, c, d], not just plane[:3]!
        heights = -heights

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

    # Given plane = [a, b, c, d] and inlier_cloud:
    z_axis = plane[:3] / np.linalg.norm(plane[:3])
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    origin = np.mean(np.asarray(inlier_cloud.points), axis=0)

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = origin

    frame.transform(T)  # 4x4 matrix with [x, y, z, origin]

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, frame],
                                      front=viewpoint['trajectory'][0]['front'],
                                      lookat=viewpoint['trajectory'][0]['lookat'],
                                      up=viewpoint['trajectory'][0]['up']
                                      )

    # --- Step 2: Height Slice & Tabletop Cluster ---
    # TODO: Select points with 0.5 <= h_floor(p) <= 1.0
    # TODO: Run segment_plane(...) on the height slice to find the table plane
    # TODO: Run cluster_dbscan(...) on the table-plane inliers and keep only the largest cluster

    slice_idx = np.where((heights >= 0.5) & (heights <= 1.0))[0]
    slice_cloud = outlier_cloud.select_by_index(slice_idx)

    table_plane, table_inliers = slice_cloud.segment_plane(
        distance_threshold=0.02,  # Thickness tolerance (meters) around the fitted plane
        ransac_n=3,  # Number of points to sample per trial (keep 3 for planes)
        num_iterations=1000  # Number of random trials
    )
    table_plane_cloud = slice_cloud.select_by_index(table_inliers)
    table_plane_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # green

    rest_of_slice = slice_cloud.select_by_index(table_inliers, invert=True)

    o3d.visualization.draw_geometries([table_plane_cloud, rest_of_slice],
                                      front=viewpoint['trajectory'][0]['front'],
                                      lookat=viewpoint['trajectory'][0]['lookat'],
                                      up=viewpoint['trajectory'][0]['up']
                                      )

    # --- Step 3: Table Reference Frame & Transformation ---
    # TODO: Compute tabletop centroid and ensure table_normal points away from the floor (dot(n_table, n_floor) > 0)
    # TODO: Build T_scene_table = build_table_frame(table_normal, table_center)
    # TODO: Transform scene points into the table frame using np.linalg.inv(T_scene_table)

    # --- Step 4: Tabletop Volume Crop & Object Clustering ---
    # TODO: Get [x_min, x_max] and [y_min, y_max] from the tabletop points in the table frame
    # TODO: Keep points inside x/y bounds with 0.01 <= z <= 0.40
    # TODO: Run cluster_dbscan(...) to separate individual objects and visualize


if __name__ == "__main__":
    main()
