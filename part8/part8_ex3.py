#! /usr/bin/env python3

import copy
import json
import numpy as np
import open3d as o3d


def draw_registration_result(source, target, transformation, name='open3d'):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=str(name))


def draw_all_registration_results(source, target, results):
    visualizers = []
    for idx, (label, res) in enumerate(results):
        src_temp = copy.deepcopy(source)
        tgt_temp = copy.deepcopy(target)
        src_temp.paint_uniform_color([1, 0.706, 0])
        tgt_temp.paint_uniform_color([0, 0.651, 0.929])
        src_temp.transform(res.transformation)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=label, width=420, height=420, left=50 + idx * 430, top=100)
        vis.add_geometry(src_temp)
        vis.add_geometry(tgt_temp)
        visualizers.append(vis)

    while visualizers:
        still_open = []
        for vis in visualizers:
            if vis.poll_events():
                vis.update_renderer()
                still_open.append(vis)
            else:
                vis.destroy_window()
        visualizers = still_open


def compute_registration_error(T_est: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    E = np.linalg.inv(T_true) @ T_est
    trans_err_mm = float(np.linalg.norm(E[:3, 3]) * 1000.0)
    cos_theta = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(cos_theta)))
    return trans_err_mm, rot_err_deg


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    # 1: Downsample with pcd.voxel_down_sample(voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    # 2: Estimate normals on pcd_down (radius = voxel_size * 2.0, max_nn = 30)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    # 3: Compute FPFH features with o3d.pipelines.registration.compute_fpfh_feature(...)
    #         (radius = voxel_size * 5.0, max_nn = 100)
    pcd_down.orient_normals_consistent_tangent_plane(30)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))

    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size: float):
    distance_threshold = voxel_size * 1.5
    # 4: Run o3d.pipelines.registration.registration_ransac_based_on_feature_matching(...)
    #         with mutual_filter=True, ransac_n=3,
    #         TransformationEstimationPointToPoint(False),
    #         CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #         CorrespondenceCheckerBasedOnDistance(distance_threshold),
    #         and RANSACConvergenceCriteria(100000, 0.999)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result



def main():
    # Load ground-truth matrix
    with open("icp_point_clouds/T_true.json", "r") as f:
        T_true = np.array(json.load(f)["T_part_1_part_2"], dtype=np.float64)

    # Load point clouds (part_2 -> source, part_1 -> target)
    source = o3d.io.read_point_cloud("icp_point_clouds/part_2.ply")
    target = o3d.io.read_point_cloud("icp_point_clouds/part_1.ply")

    # Estimate normals on full-resolution clouds for ICP refinement
    for pcd in (source, target):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)

    # --- Step 1: Global Registration (FPFH + RANSAC) ---
    o3d.utility.random.seed(42)
    voxel_size = 0.05
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    print("RANSAC Global Registration:", result_ransac)

    # --- Step 2: Local ICP Refinement (using RANSAC output as trans_init) ---
    trans_init = result_ransac.transformation
    threshold = voxel_size * 2

    reg_p2p_coarse = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 0.02, reg_p2p_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )

    reg_p2l_coarse = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, 0.02, reg_p2l_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )

    reg_sym_coarse = o3d.pipelines.registration.registration_symmetric_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationSymmetric(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    reg_sym = o3d.pipelines.registration.registration_symmetric_icp(
        source, target, 0.02, reg_sym_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationSymmetric(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )

    # --- Step 3: Compare RANSAC vs. ICP Refinement against T_true.json ---
    print("\n=== Error Comparison vs T_true.json (evaluated at 0.02 m) ===")
    results = [
        ("RANSAC Global", result_ransac),
        ("Point-to-Point ICP", reg_p2p),
        ("Point-to-Plane ICP", reg_p2l),
        ("Symmetric ICP", reg_sym),
    ]
    for label, res in results:
        eval_res = o3d.pipelines.registration.evaluate_registration(
            source, target, 0.02, res.transformation
        )
        t_err, r_err = compute_registration_error(res.transformation, T_true)
        print(
            f"{label:<20} | fitness: {eval_res.fitness:.6f} | inlier_rmse: {eval_res.inlier_rmse:.6f} | "
            f"{t_err:.2f} mm | {r_err:.2f} deg"
        )

    draw_all_registration_results(source, target, results)


if __name__ == "__main__":
    main()
