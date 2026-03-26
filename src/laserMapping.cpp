// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <pcl/filters/crop_box.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
mutex mtx_lidar2_filtered;  // 保护 lidar2_filtered_ptr，供回调写入、timer 发布
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
string lid_topic_2;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;
bool   multi_en = false; // whether to use multiple lidars
bool   debug_info = false; // whether to print debug info
bool   removed_en = false; // whether to publish removed points

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);

vector<double>       Lidar2_to_Lidar_T(3, 0.0);
vector<double>       Lidar2_to_Lidar_R(9, 0.0);
Eigen::Matrix<double, 3, 3> Lidar2_R_wrt_Lidar;
Eigen::Matrix<double, 3, 1> Lidar2_T_wrt_Lidar;
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer_2; // second lidar buffer
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
PointCloudXYZI::Ptr lidar2_filtered_ptr(new PointCloudXYZI());
PointCloudXYZI::Ptr lidar2_removed_cloud;  // 第二雷达被滤掉的点云（后腿等），用于调试发布

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
/** 多雷达时第二路独立 preprocess，可与主雷达 lidar_type / 线数等不同 */
shared_ptr<Preprocess> p_pre_2(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

static inline bool any_lidar_is_airy()
{
    if (p_pre->lidar_type == AIRY) return true;
    if (multi_en && p_pre_2->lidar_type == AIRY) return true;
    return false;
}

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void filter_rear_legs_lidar1(const PointCloudXYZI::Ptr& input,
    PointCloudXYZI::Ptr& output,
    PointCloudXYZI::Ptr removed_out)
{
    if (!input || input->empty()) {
        if (output) output->clear();
        if (removed_out) removed_out->clear();
        return;
    }

    // 确保 PCL 滤波器能正确遍历（部分版本依赖 width/height）
    PointCloudXYZI::Ptr input_fixed(new PointCloudXYZI());
    *input_fixed = *input;
    if (input_fixed->width == 0 && !input_fixed->empty()) {
        input_fixed->width = input_fixed->size();
        input_fixed->height = 1;
    }

    PointCloudXYZI::Ptr tmp(new PointCloudXYZI());
    PointCloudXYZI::Ptr tmp2(new PointCloudXYZI());

    // 主雷达坐标系下后腿大致范围（x≈高度, y≈左右, z≈前后，后腿在 z 正方向）
    // 适当放宽边界以免漏掉边缘点
    const float margin = 0.02f;
    // ===== 左后腿 RL (y 正) =====
    // 内侧 (y_min) 不加 margin，避免两个盒子在 y=0 处连成一片
    pcl::CropBox<PointType> crop_rl;
    crop_rl.setInputCloud(input_fixed);
    crop_rl.setMin(Eigen::Vector4f(-0.05f - margin,  0.15f,          -1.1f - margin, 1.0f));
    crop_rl.setMax(Eigen::Vector4f( 0.82f + margin,  0.30f + margin, -0.80f + margin, 1.0f));
    crop_rl.setNegative(true);
    crop_rl.filter(*tmp);

    // ===== 右后腿 RR (y 负) =====
    // 内侧 (y_max) 不加 margin，同理
    pcl::CropBox<PointType> crop_rr;
    crop_rr.setInputCloud(tmp);
    crop_rr.setMin(Eigen::Vector4f(-0.05f - margin, -0.30f - margin, -1.1f - margin, 1.0f));
    crop_rr.setMax(Eigen::Vector4f( 0.82f + margin, -0.15f,          -0.80f + margin, 1.0f));
    crop_rr.setNegative(true);
    crop_rr.filter(*tmp2);

    *output = *tmp2;

    // 若需要，收集被滤掉的点（两个盒子内的点）并输出
    if (removed_en) {
        removed_out->clear();
        PointCloudXYZI::Ptr in_rl(new PointCloudXYZI());
        PointCloudXYZI::Ptr in_rr(new PointCloudXYZI());
        pcl::CropBox<PointType> box_rl, box_rr;
        box_rl.setInputCloud(input_fixed);
        box_rl.setMin(Eigen::Vector4f(-0.05f - margin,  0.04f,          -1.1f - margin, 1.0f));
        box_rl.setMax(Eigen::Vector4f( 0.82f + margin,  0.30f + margin, -0.70f + margin, 1.0f));
        box_rl.setNegative(false);
        box_rl.filter(*in_rl);
        box_rr.setInputCloud(input_fixed);
        box_rr.setMin(Eigen::Vector4f(-0.05f - margin, -0.30f - margin, -1.1f - margin, 1.0f));
        box_rr.setMax(Eigen::Vector4f( 0.82f + margin, -0.04f,          -0.70f + margin, 1.0f));
        box_rr.setNegative(false);
        box_rr.filter(*in_rr);
        *removed_out = *in_rl;
        *removed_out += *in_rr;
    }

    // 调试：若滤掉不少点则打印一次（避免刷屏）
    if (removed_en) {
    static int _dbg_count = 0;
    if (++_dbg_count % 50 == 0) {
        size_t in_sz = input->size(), out_sz = tmp2->size();
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
            "[filter_rear_legs] lidar2 in=%zu out=%zu removed=%zu",
                in_sz, out_sz, in_sz > out_sz ? in_sz - out_sz : 0);
        }
    }
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping"), "Standard PCL callback");
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "time diff: %f", cur_time - last_timestamp_lidar);
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    // RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "here is ok");
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "time diff: %f", cur_time - last_timestamp_lidar);
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

 void livox_pcl_cbk2(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
 {
    RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping"), "Lidar 2 callback");

    // 1. 点云处理与外参转换
    
    // 构造输出点云
    PointCloudXYZI::Ptr lidar2_ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr filtered_ptr(new PointCloudXYZI());
    p_pre_2->process(msg, lidar2_ptr);

    // 利用外参 (Lidar2_R_wrt_Lidar, Lidar2_T_wrt_Lidar) 将点云从雷达2坐标系变换到主雷达坐标系
    for (auto& point : lidar2_ptr->points) {
        Eigen::Vector3d pt_lidar2(point.x, point.y, point.z);
        Eigen::Vector3d pt_lidar1 = Lidar2_R_wrt_Lidar * pt_lidar2 + Lidar2_T_wrt_Lidar;
        point.x = pt_lidar1(0);
        point.y = pt_lidar1(1);
        point.z = pt_lidar1(2);
    }

    // 预分配空间（避免vector反复扩容）
    filtered_ptr->points.reserve(lidar2_ptr->points.size());

    PointCloudXYZI::Ptr removed_ptr(new PointCloudXYZI());
    filter_rear_legs_lidar1(lidar2_ptr, filtered_ptr, removed_ptr);

    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        lidar_buffer_2.push_back(filtered_ptr);
        lidar2_removed_cloud = removed_ptr;
    }

    
 }

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping"), "IMU callback");
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/** 第二路 PointCloud2（Velodyne / Ouster / MID360 等非 Livox CustomMsg） */
void standard_pcl_cbk2(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping"), "Lidar 2 standard PCL callback");

    PointCloudXYZI::Ptr lidar2_ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr filtered_ptr(new PointCloudXYZI());
    p_pre_2->process(msg, lidar2_ptr);

    for (auto& point : lidar2_ptr->points) {
        Eigen::Vector3d pt_lidar2(point.x, point.y, point.z);
        Eigen::Vector3d pt_lidar1 = Lidar2_R_wrt_Lidar * pt_lidar2 + Lidar2_T_wrt_Lidar;
        point.x = pt_lidar1(0);
        point.y = pt_lidar1(1);
        point.z = pt_lidar1(2);
    }

    filtered_ptr->points.reserve(lidar2_ptr->points.size());
    PointCloudXYZI::Ptr removed_ptr(new PointCloudXYZI());
    filter_rear_legs_lidar1(lidar2_ptr, filtered_ptr, removed_ptr);

    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        lidar_buffer_2.push_back(filtered_ptr);
        lidar2_removed_cloud = removed_ptr;
    }
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping"), "sync_packages");
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}



PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body_fusion(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_fusion)
{
    // 1. 主雷达当前帧（主雷达坐标系）
    int size_main = feats_undistort->points.size();

    // 2. 取第二雷达最新一帧（已经在 livox_pcl_cbk2 中变到主雷达坐标系）
    PointCloudXYZI::Ptr lidar2_cloud(new PointCloudXYZI());
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        if (!lidar_buffer_2.empty()) {
            lidar2_cloud = lidar_buffer_2.back();
        }
    }

    // 3. 先在主雷达坐标系下融合
    PointCloudXYZI::Ptr fusion_lidar(new PointCloudXYZI());
    fusion_lidar->reserve(size_main + (lidar2_cloud ? lidar2_cloud->size() : 0));

    if (!feats_undistort->empty()) {
        *fusion_lidar += *feats_undistort;
    }

    if (lidar2_cloud && !lidar2_cloud->empty()) {
        *fusion_lidar += *lidar2_cloud;
    }

    if (fusion_lidar->empty()) {
        return;
    }

    // 4. 再统一从主雷达坐标系 -> IMU/body 坐标系
    PointCloudXYZI::Ptr fusion_body(new PointCloudXYZI(fusion_lidar->size(), 1));
    for (size_t i = 0; i < fusion_lidar->points.size(); ++i)
    {
        RGBpointBodyLidarToIMU(&fusion_lidar->points[i], &fusion_body->points[i]);
    }
    // 5. 发布
    sensor_msgs::msg::PointCloud2 laserCloudMsg;
    pcl::toROSMsg(*fusion_body, laserCloudMsg);
    laserCloudMsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudMsg.header.frame_id = "_body";
    pubLaserCloudFull_fusion->publish(laserCloudMsg);
}

void publish_lidar2_removed(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_lidar2_removed)
{
    if (!pub_lidar2_removed) return;
    PointCloudXYZI::Ptr removed;
    {
        std::lock_guard<std::mutex> lock(mtx_buffer);
        removed = lidar2_removed_cloud;
    }
    if (!removed || removed->empty()) return;
    PointCloudXYZI::Ptr removed_body(new PointCloudXYZI(removed->size(), 1));
    for (size_t i = 0; i < removed->points.size(); ++i)
        RGBpointBodyLidarToIMU(&removed->points[i], &removed_body->points[i]);
    sensor_msgs::msg::PointCloud2 msg_removed;
    pcl::toROSMsg(*removed_body, msg_removed);
    msg_removed.header.stamp = get_ros_time(lidar_end_time);
    msg_removed.header.frame_id = "_body";
    pub_lidar2_removed->publish(msg_removed);
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = any_lidar_is_airy() ? "rslidar" : "_body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = "camera_init";
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "robot";
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);

    // 与 TF body_yaw、robot 一致：R_camera_init_robot = R_ci_body_yaw * R_body_yaw_robot
    double body_yaw_in_camera_init = 0.0;
    double pitch = 0.0;
    {
        double qw = geoQuat.w;
        double qx = geoQuat.x;
        double qy = geoQuat.y;
        double qz = geoQuat.z;
        double sinr_cosp = 2.0 * (qw * qx + qy * qz);
        double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
        body_yaw_in_camera_init = std::atan2(sinr_cosp, cosr_cosp);
        double sinp = 2.0 * (qw * qy - qz * qx);
        if (std::abs(sinp) >= 1)
            pitch = std::copysign(M_PI / 2, sinp);
        else
            pitch = std::asin(sinp);
        if (std::cos(body_yaw_in_camera_init) < 0)
            pitch = -pitch;
    }
    const double offset_x = -0.5 * sin(pitch);
    const double offset_z = -0.5 * cos(pitch);

    tf2::Quaternion q_ci_by;
    q_ci_by.setRPY(body_yaw_in_camera_init, 0.0, 0.0);
    q_ci_by.normalize();
    tf2::Matrix3x3 m_ci_by(q_ci_by);
    Eigen::Matrix3d R_ci_by;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R_ci_by(r, c) = m_ci_by[r][c];

    tf2::Quaternion q_by_ro(0.0, -0.70710678, 0.0, 0.70710678);
    tf2::Matrix3x3 m_by_ro(q_by_ro);
    Eigen::Matrix3d R_by_ro;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R_by_ro(r, c) = m_by_ro[r][c];

    const Eigen::Matrix3d R_ci_ro = R_ci_by * R_by_ro;
    const Eigen::Vector3d t_by_ro(offset_x, 0.0, offset_z);
    const Eigen::Vector3d p_imu(state_point.pos(0), state_point.pos(1), state_point.pos(2));
    const Eigen::Vector3d p_robot_ci = p_imu + R_ci_by * t_by_ro;

    odomAftMapped.pose.pose.position.x = p_robot_ci(0);
    odomAftMapped.pose.pose.position.y = p_robot_ci(1);
    odomAftMapped.pose.pose.position.z = p_robot_ci(2);
    {
        Eigen::Quaterniond q_ci_ro(R_ci_ro);
        q_ci_ro.normalize();
        odomAftMapped.pose.pose.orientation.x = q_ci_ro.x();
        odomAftMapped.pose.pose.orientation.y = q_ci_ro.y();
        odomAftMapped.pose.pose.orientation.z = q_ci_ro.z();
        odomAftMapped.pose.pose.orientation.w = q_ci_ro.w();
    }

    // 局部速度：robot 原点在世界系下的速度，再投影到 robot 轴（非世界系分量）
    const Eigen::Matrix3d R_wb = state_point.rot.toRotationMatrix();
    Eigen::Vector3d omega_b(0.0, 0.0, 0.0);
    if (!Measures.imu.empty())
    {
        const auto &im = Measures.imu.back();
        omega_b << im->angular_velocity.x - state_point.bg(0),
            im->angular_velocity.y - state_point.bg(1),
            im->angular_velocity.z - state_point.bg(2);
    }
    const Eigen::Vector3d omega_w = R_wb * omega_b;
    const Eigen::Vector3d delta_w = R_ci_by * t_by_ro;
    const Eigen::Vector3d v_imu_w(state_point.vel(0), state_point.vel(1), state_point.vel(2));
    const Eigen::Vector3d v_robot_w = v_imu_w + omega_w.cross(delta_w);
    const Eigen::Vector3d v_robot = R_ci_ro.transpose() * v_robot_w;
    const Eigen::Vector3d omega_robot = R_ci_ro.transpose() * omega_w;

    odomAftMapped.twist.twist.linear.x = v_robot(0);
    odomAftMapped.twist.twist.linear.y = v_robot(1);
    odomAftMapped.twist.twist.linear.z = v_robot(2);
    odomAftMapped.twist.twist.angular.x = omega_robot(0);
    odomAftMapped.twist.twist.angular.y = omega_robot(1);
    odomAftMapped.twist.twist.angular.z = omega_robot(2);

    // RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "v_robot_x: %f",  v_robot(0));
    // RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "v_robot_y: %f",  v_robot(1));
    // RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "v_robot_z: %f",  v_robot(2));


    auto P = kf.get_P();
    Eigen::Matrix3d P_vel_world = P.block<3, 3>(12, 12);
    Eigen::Matrix3d P_vel_robot = R_ci_ro.transpose() * P_vel_world * R_ci_ro;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            odomAftMapped.twist.covariance[i * 6 + j] = P_vel_robot(i, j);

    const double gyr_var = gyr_cov * gyr_cov;
    for (int i = 3; i < 6; i++)
        odomAftMapped.twist.covariance[i * 6 + i] = gyr_var;

    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }
    pubOdomAftMapped->publish(odomAftMapped);

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.child_frame_id = "_body";
    trans.header.stamp = get_ros_time(lidar_end_time);
    trans.transform.translation.x = state_point.pos(0);
    trans.transform.translation.y = state_point.pos(1);
    trans.transform.translation.z = state_point.pos(2);
    trans.transform.rotation.w = geoQuat.w;
    trans.transform.rotation.x = geoQuat.x;
    trans.transform.rotation.y = geoQuat.y;
    trans.transform.rotation.z = geoQuat.z;
    tf_br->sendTransform(trans);

    // RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "pitch: %f", pitch);

    geometry_msgs::msg::TransformStamped trans_body_yaw;
    trans_body_yaw.header.frame_id = "camera_init";
    trans_body_yaw.child_frame_id = "body_yaw";
    trans_body_yaw.header.stamp = get_ros_time(lidar_end_time);
    trans_body_yaw.transform.translation.x = state_point.pos(0);
    trans_body_yaw.transform.translation.y = state_point.pos(1);
    trans_body_yaw.transform.translation.z = state_point.pos(2);
    trans_body_yaw.transform.rotation.x = q_ci_by.x();
    trans_body_yaw.transform.rotation.y = q_ci_by.y();
    trans_body_yaw.transform.rotation.z = q_ci_by.z();
    trans_body_yaw.transform.rotation.w = q_ci_by.w();
    tf_br->sendTransform(trans_body_yaw);

    // RCLCPP_INFO(rclcpp::get_logger("laser_mapping"), "offset_x: %f, offset_z: %f", offset_x, offset_z);
    // 发布 body_yaw -> robot
    geometry_msgs::msg::TransformStamped trans_body_yaw_robot;
    trans_body_yaw_robot.header.frame_id = "body_yaw";
    trans_body_yaw_robot.child_frame_id = "robot";
    trans_body_yaw_robot.header.stamp = get_ros_time(lidar_end_time);
    trans_body_yaw_robot.transform.translation.x = offset_x;
    trans_body_yaw_robot.transform.translation.y = 0.0;
    trans_body_yaw_robot.transform.translation.z = offset_z;
    trans_body_yaw_robot.transform.rotation.w = 0.70710678;
    trans_body_yaw_robot.transform.rotation.x = 0.0;
    trans_body_yaw_robot.transform.rotation.y = -0.70710678;
    trans_body_yaw_robot.transform.rotation.z = 0.0;
    tf_br->sendTransform(trans_body_yaw_robot);
    
    // 仅 Airy：body -> rslidar，用 Lidar-IMU 外参把 body 下的点云恢复到雷达“正”的坐标系
    if (any_lidar_is_airy())
    {
        geometry_msgs::msg::TransformStamped trans_rslidar;
        trans_rslidar.header.frame_id = "_body";
        trans_rslidar.child_frame_id = "rslidar";
        trans_rslidar.header.stamp = get_ros_time(lidar_end_time);
        trans_rslidar.transform.translation.x = state_point.offset_T_L_I(0);
        trans_rslidar.transform.translation.y = state_point.offset_T_L_I(1);
        trans_rslidar.transform.translation.z = state_point.offset_T_L_I(2);
        trans_rslidar.transform.rotation.x = state_point.offset_R_L_I.coeffs()[0];
        trans_rslidar.transform.rotation.y = state_point.offset_R_L_I.coeffs()[1];
        trans_rslidar.transform.rotation.z = state_point.offset_R_L_I.coeffs()[2];
        trans_rslidar.transform.rotation.w = state_point.offset_R_L_I.coeffs()[3];
        tf_br->sendTransform(trans_rslidar);
    }
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<bool>("common.debug_info", false);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter<bool>("multi.multi_en", false);
        this->declare_parameter<string>("multi.lid_topic_2", "/livox/lidar_2");
        this->declare_parameter<vector<double>>("multi.Lidar2_to_Lidar_T", vector<double>());
        this->declare_parameter<vector<double>>("multi.Lidar2_to_Lidar_R", vector<double>());
        this->declare_parameter<bool>("multi.removed_en", false);
        this->declare_parameter<int>("multi.lidar_type", -1);   // -1: 与 preprocess.lidar_type 相同
        this->declare_parameter<int>("multi.scan_line", -1);    // -1: 与主雷达 N_SCANS 相同
        this->declare_parameter<int>("multi.timestamp_unit", -1); // -1: 与主雷达 time_unit 相同
        this->declare_parameter<int>("multi.scan_rate", -1);     // -1: 与主雷达 SCAN_RATE 相同

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<bool>("common.debug_info", debug_info, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<bool>("multi.multi_en", multi_en, false);    
        this->get_parameter_or<string>("multi.lid_topic_2", lid_topic_2, "/livox/lidar_2");
        this->get_parameter_or<vector<double>>("multi.Lidar2_to_Lidar_T", Lidar2_to_Lidar_T, vector<double>());
        this->get_parameter_or<vector<double>>("multi.Lidar2_to_Lidar_R", Lidar2_to_Lidar_R, vector<double>());
        this->get_parameter_or<bool>("multi.removed_en", removed_en, false);

        // 第二路 preprocess：默认与主雷达一致，multi.* 可单独覆盖（支持异构雷达）
        p_pre_2->blind = p_pre->blind;
        p_pre_2->point_filter_num = p_pre->point_filter_num;
        p_pre_2->N_SCANS = p_pre->N_SCANS;
        p_pre_2->SCAN_RATE = p_pre->SCAN_RATE;
        p_pre_2->time_unit = p_pre->time_unit;
        p_pre_2->feature_enabled = p_pre->feature_enabled;
        p_pre_2->given_offset_time = p_pre->given_offset_time;
        {
            int multi_lidar_type = -1;
            this->get_parameter_or<int>("multi.lidar_type", multi_lidar_type, -1);
            p_pre_2->lidar_type = (multi_lidar_type >= 0) ? multi_lidar_type : p_pre->lidar_type;
            int multi_scan_line = -1;
            this->get_parameter_or<int>("multi.scan_line", multi_scan_line, -1);
            if (multi_scan_line > 0) p_pre_2->N_SCANS = multi_scan_line;
            int multi_tu = -1;
            this->get_parameter_or<int>("multi.timestamp_unit", multi_tu, -1);
            if (multi_tu >= 0) p_pre_2->time_unit = multi_tu;
            int multi_sr = -1;
            this->get_parameter_or<int>("multi.scan_rate", multi_sr, -1);
            if (multi_sr > 0) p_pre_2->SCAN_RATE = multi_sr;
        }

        if (debug_info)
        {
            // 彩色输出定义，便于 debug
            const char* COLOR_RESET   = "\033[0m";
            const char* COLOR_HEAD    = "\033[1;36m";
            const char* COLOR_ITEM    = "\033[0;33m";
            const char* COLOR_BOOL_ON = "\033[1;32m";
            const char* COLOR_BOOL_OFF= "\033[1;31m";
            const char* COLOR_STRING  = "\033[1;37m";
            const char* COLOR_NUMBER  = "\033[1;36m";

            RCLCPP_INFO(this->get_logger(), "%s============== FAST_LIO2 Loaded Parameters ==============%s", COLOR_HEAD, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%spath_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, path_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, path_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%seffect_pub_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, effect_pub_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, effect_pub_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%smap_pub_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, map_pub_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, map_pub_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sscan_pub_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, scan_pub_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, scan_pub_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sdense_pub_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, dense_pub_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, dense_pub_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sscan_body_pub_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, scan_body_pub_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, scan_body_pub_en ? "true" : "false", COLOR_RESET);       
            RCLCPP_INFO(this->get_logger(), "%sNUM_MAX_ITERATIONS%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, NUM_MAX_ITERATIONS, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%smap_file_path%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, map_file_path.c_str(), COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%slid_topic%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, lid_topic.c_str(), COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%simu_topic%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, imu_topic.c_str(), COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%stime_sync_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, time_sync_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, time_sync_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%stime_diff_lidar_to_imu%s: %s%.9f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, time_diff_lidar_to_imu, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sfilter_size_corner_min%s: %s%.3f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, filter_size_corner_min, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sfilter_size_surf_min%s: %s%.3f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, filter_size_surf_min, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sfilter_size_map_min%s: %s%.3f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, filter_size_map_min, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%scube_len%s: %s%.3f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, cube_len, COLOR_RESET);    
            RCLCPP_INFO(this->get_logger(), "%sDET_RANGE%s: %s%.3f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, DET_RANGE, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sfov_deg%s: %s%.3f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, fov_deg, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sgyr_cov%s: %s%.6f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, gyr_cov, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sacc_cov%s: %s%.6f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, acc_cov, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sb_gyr_cov%s: %s%.6f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, b_gyr_cov, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sb_acc_cov%s: %s%.6f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, b_acc_cov, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sp_pre->blind%s: %s%.3f%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre->blind, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sp_pre->lidar_type%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre->lidar_type, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sp_pre->N_SCANS%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre->N_SCANS, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sp_pre->time_unit%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre->time_unit, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sp_pre->SCAN_RATE%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre->SCAN_RATE, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sp_pre->point_filter_num%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre->point_filter_num, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sp_pre->feature_enabled%s: %s%s%s", COLOR_ITEM, COLOR_RESET, p_pre->feature_enabled ? COLOR_BOOL_ON : COLOR_BOOL_OFF, p_pre->feature_enabled ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sruntime_pos_log%s: %s%s%s", COLOR_ITEM, COLOR_RESET, runtime_pos_log ? COLOR_BOOL_ON : COLOR_BOOL_OFF, runtime_pos_log ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sextrinsic_est_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, extrinsic_est_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, extrinsic_est_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%spcd_save_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, pcd_save_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, pcd_save_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%spcd_save_interval%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, pcd_save_interval, COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%smulti_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, multi_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, multi_en ? "true" : "false", COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%slid_topic_2%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, lid_topic_2.c_str(), COLOR_RESET);
            RCLCPP_INFO(this->get_logger(), "%sremoved_en%s: %s%s%s", COLOR_ITEM, COLOR_RESET, removed_en ? COLOR_BOOL_ON : COLOR_BOOL_OFF, removed_en ? "true" : "false", COLOR_RESET);
            if (multi_en)
            {
                RCLCPP_INFO(this->get_logger(), "%sp_pre_2->lidar_type%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre_2->lidar_type, COLOR_RESET);
                RCLCPP_INFO(this->get_logger(), "%sp_pre_2->N_SCANS%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre_2->N_SCANS, COLOR_RESET);
                RCLCPP_INFO(this->get_logger(), "%sp_pre_2->time_unit%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre_2->time_unit, COLOR_RESET);
                RCLCPP_INFO(this->get_logger(), "%sp_pre_2->SCAN_RATE%s: %s%d%s", COLOR_ITEM, COLOR_RESET, COLOR_NUMBER, p_pre_2->SCAN_RATE, COLOR_RESET);
            }
            

            // 打印 extrinT 和 extrinR
            std::ostringstream ssT, ssR;
            std::stringstream ssT2, ssR2;
            ssT << "[";
            for(size_t i=0; i<extrinT.size(); ++i){
                ssT << extrinT[i];
                if(i!=extrinT.size()-1) ssT << ", ";
            }
            ssT << "]";
            RCLCPP_INFO(this->get_logger(), "%sextrinT%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, ssT.str().c_str(), COLOR_RESET);

            ssR << "[";
            for(size_t i=0; i<extrinR.size(); ++i){
                ssR << extrinR[i];
                if(i!=extrinR.size()-1) ssR << ", ";
            }
            ssR << "]";
            RCLCPP_INFO(this->get_logger(), "%sextrinR%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, ssR.str().c_str(), COLOR_RESET);

            ssT2 << "[";
            for(size_t i=0; i<Lidar2_to_Lidar_T.size(); ++i){
                ssT2 << Lidar2_to_Lidar_T[i];
                if(i!=Lidar2_to_Lidar_T.size()-1) ssT2 << ", ";
            }
            ssT2 << "]";
            RCLCPP_INFO(this->get_logger(), "%sLidar2_to_Lidar_T%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, ssT.str().c_str(), COLOR_RESET);
            
            ssR2 << "[";
            for(size_t i=0; i<Lidar2_to_Lidar_R.size(); ++i){
                ssR << Lidar2_to_Lidar_R[i];
                if(i!=Lidar2_to_Lidar_R.size()-1) ssR << ", ";
            }
            ssR2 << "]";
            RCLCPP_INFO(this->get_logger(), "%sLidar2_to_Lidar_R%s: %s%s%s", COLOR_ITEM, COLOR_RESET, COLOR_STRING, ssR.str().c_str(), COLOR_RESET);

            RCLCPP_INFO(this->get_logger(), "%s============== END PARAMS ==============%s", COLOR_HEAD, COLOR_RESET);
        }

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="camera_init";

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        Lidar2_R_wrt_Lidar = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(Lidar2_to_Lidar_R.data());
        Lidar2_T_wrt_Lidar = Eigen::Map<Eigen::Matrix<double, 3, 1>>(Lidar2_to_Lidar_T.data());

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
        // Callback Group
        imu_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        pcl_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        pcl_livoxcallback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        pcl_livoxcallback_group_2 = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        timer_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions pcl_options;
            pcl_options.callback_group = pcl_callback_group_;
        rclcpp::SubscriptionOptions pcl_options_2;
            pcl_options_2.callback_group = pcl_callback_group_2;
        rclcpp::SubscriptionOptions pcl_livox_options;
            pcl_livox_options.callback_group = pcl_livoxcallback_group_;
        rclcpp::SubscriptionOptions pcl_livox_options_2;
            pcl_livox_options_2.callback_group = pcl_livoxcallback_group_2;
        rclcpp::SubscriptionOptions imu_options;
            imu_options.callback_group = imu_callback_group_;
        // rclcpp::SubscriptionOptions timer_options;
        //     timer_options.callback_group = timer_callback_group_;

        if (multi_en)
        {
            if (p_pre_2->lidar_type == AVIA)
            {
                sub_pcl_livox_2_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                    lid_topic_2, 20, livox_pcl_cbk2, pcl_livox_options_2);
            }
            else
            {
                sub_pcl_pc_2_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                    lid_topic_2, rclcpp::SensorDataQoS(), standard_pcl_cbk2, pcl_options_2);
            }
        }
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk, pcl_livox_options);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk, pcl_options);
        }

        
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, imu_cbk, imu_options);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
        {
            geometry_msgs::msg::TransformStamped base_to_camera_init;
            base_to_camera_init.header.stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
            base_to_camera_init.header.frame_id = "base";
            base_to_camera_init.child_frame_id = "camera_init";
            base_to_camera_init.transform.translation.x = 0.5;
            base_to_camera_init.transform.translation.y = 0.0;
            base_to_camera_init.transform.translation.z = 0.0;
            base_to_camera_init.transform.rotation.w = 0.70710678;
            base_to_camera_init.transform.rotation.x = 0.0;
            base_to_camera_init.transform.rotation.y = 0.70710678;
            base_to_camera_init.transform.rotation.z = 0.0;
            static_tf_broadcaster_->sendTransform(base_to_camera_init);
        }
        pubLaserCloudFull_fusion_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_lidar2_filtered", 20);  // 测试用：第二雷达裁剪+外参变换后的点云，在 timer 中发布
        pubLaserCloudRemoved_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_lidar2_removed", 20);      // 被滤掉的点云（后腿等），调试用
        

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this), timer_callback_group_);

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

private:
    void timer_callback()
    {
        RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping"), "timer_callback");
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                RCLCPP_INFO(this->get_logger(), "This is first scan, first_lidar_time %f", first_lidar_time);
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // 第二雷达裁剪+外参变换后的点云（测试过滤），与其它点云一起在 timer 中发布
            if (multi_en) {
                publish_frame_body_fusion(pubLaserCloudFull_fusion_);
                if (removed_en)
                    publish_lidar2_removed(pubLaserCloudRemoved_);
            }
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en) publish_map(pubLaserCloudMap_);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_fusion_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudRemoved_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_2_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_2_;

    // Callback Group
    rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
    rclcpp::CallbackGroup::SharedPtr pcl_callback_group_;
    rclcpp::CallbackGroup::SharedPtr pcl_callback_group_2;
    rclcpp::CallbackGroup::SharedPtr pcl_livoxcallback_group_;
    rclcpp::CallbackGroup::SharedPtr pcl_livoxcallback_group_2;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    rclcpp::spin(std::make_shared<LaserMappingNode>());

    // // 启用多线程执行器
    // rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    // auto node = std::make_shared<LaserMappingNode>();
    // executor.add_node(node);
    // executor.spin();

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
