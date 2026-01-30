# FAST-LIO2
**说明：** 本 Fork 针对 **Airy** 雷达进行了适配。针对 Airy 的硬件安装方式对坐标系变换进行了适配，同时保留了模块化设计，可通过配置适配不同的雷达。

**[English](README.md)**

---



# 原版本
> ROS2 Fork 仓库维护者：[Ericsiii](https://github.com/Ericsii)

## 相关工作与扩展应用

**SLAM：**

1. [ikd-Tree](https://github.com/hku-mars/ikd-Tree)：用于 3D kNN 搜索的动态 KD-Tree。
2. [R2LIVE](https://github.com/hku-mars/r2live)：以 FAST-LIO 作为 LiDAR-IMU 前端的 LiDAR-惯性-视觉融合方案。
3. [LI_Init](https://github.com/hku-mars/LiDAR_IMU_Init)：鲁棒的实时 LiDAR-IMU 外参初始化与同步包。
4. [FAST-LIO-LOCALIZATION](https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION)：集成重定位功能的 FAST-LIO。

**控制与规划：**

1. [IKFOM](https://github.com/hku-mars/IKFoM)：流形上快速高精度卡尔曼滤波工具箱。
2. [UAV Avoiding Dynamic Obstacles](https://github.com/hku-mars/dyn_small_obs_avoidance)：FAST-LIO 在机器人规划中的实现之一。
3. [UGV Demo](https://www.youtube.com/watch?v=wikgrQbE6Cs)：可微流形上的轨迹跟踪模型预测控制。
4. [Bubble Planner](https://arxiv.org/abs/2202.12177)：使用 Receding Corridors 规划高速平滑四旋翼轨迹。

## FAST-LIO

**FAST-LIO**（Fast LiDAR-Inertial Odometry）是计算高效、鲁棒的 LiDAR-惯性里程计包。通过紧耦合迭代扩展卡尔曼滤波融合 LiDAR 特征点与 IMU 数据，在快速运动、噪声或退化环境中实现鲁棒导航。本包解决了以下关键问题：
1. 用于里程计优化的快速迭代卡尔曼滤波；
2. 在多数平稳环境下自动初始化；
3. 并行 KD-Tree 搜索以降低计算量。

## FAST-LIO 2.0（2021-07-05 更新）

<div align="left">
<img src="doc/real_experiment2.gif" width=49.6% />
<img src="doc/ulhkwh_fastlio.gif" width = 49.6% >
</div>

**相关视频：** [FAST-LIO2](https://youtu.be/2OvjGnxszf8)、[FAST-LIO1](https://youtu.be/iYCY6T79oNU)

**流程：**
<div align="center">
<img src="doc/overview_fastlio2.svg" width=99% />
</div>

**新特性：**
1. 基于 [ikd-Tree](https://github.com/hku-mars/ikd-Tree) 的增量建图，实现更高速度与 100Hz 以上 LiDAR 频率。
2. 在原始 LiDAR 点上直接里程计（scan-to-map），可关闭特征提取，精度更高。
3. 无需特征提取，FAST-LIO2 支持多种 LiDAR（旋转式如 Velodyne、Ouster，固态如 Livox Avia、Horizon、MID-70），并易于扩展。
4. 支持外置 IMU。
5. 支持 ARM 平台（Khadas VIM3、Nvidia TX2、Raspberry Pi 4B 8G RAM）。

**相关论文：**

[FAST-LIO2: Fast Direct LiDAR-inertial Odometry](doc/Fast_LIO_2.pdf)

[FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter](https://arxiv.org/abs/2010.08196)

**贡献者**

[Wei Xu 徐威](https://github.com/XW-HKU)、[Yixi Cai 蔡逸熙](https://github.com/Ecstasy-EC)、[Dongjiao He 贺东娇](https://github.com/Joanna-HE)、[Fangcheng Zhu 朱方程](https://github.com/zfc-zfc)、[Jiarong Lin 林家荣](https://github.com/ziv-lin)、[Zheng Liu 刘政](https://github.com/Zale-Liu)、[Borong Yuan](https://github.com/borongyuan)

## 1. 环境要求

### 1.1 **Ubuntu** 与 **ROS**

**Ubuntu >= 20.04**

系统默认的 PCL 和 Eigen 即可满足 FAST-LIO 正常运行。

ROS >= Foxy（推荐 ROS-Humble）。[ROS 安装](https://docs.ros.org/en/humble/Installation.html)

### 1.2 **PCL 与 Eigen**

PCL >= 1.8，参见 [PCL 安装](https://pointclouds.org/downloads/#linux)。

Eigen >= 3.3.4，参见 [Eigen 安装](http://eigen.tuxfamily.org/index.php?title=Main_Page)。

### <span id="1.3">1.3 **livox_ros_driver2**</span>

参见 [livox_ros_driver2 安装](https://github.com/Livox-SDK/livox_ros_driver2)。

也可使用修改版 [livox_ros_driver2](https://github.com/Ericsii/livox_ros_driver2/tree/feature/use-standard-unit)。

*说明：*
- FAST-LIO 首先支持 Livox 系列 LiDAR，因此运行任何 FAST-LIO launch 前必须安装并 **source** livox_ros_driver。
- 如何 source？最简单方式是在 `~/.bashrc` 末尾添加 `source $Livox_ros_driver_dir$/devel/setup.bash`，其中 `$Livox_ros_driver_dir$` 为 livox ros driver 工作空间目录（按官方文档一般为 `ws_livox`）。

## 2. 编译

克隆仓库并使用 colcon 编译：

```bash
    cd <ros2_ws>/src  # 进入 ros2 工作空间
    git clone https://github.com/Ericsii/FAST_LIO.git --recursive
    cd ..
    rosdep install --from-paths src --ignore-src -y
    colcon build --symlink-install
    . ./install/setup.bash  # 使用 zsh 则用 setup.zsh
```

- **编译前请先 source livox_ros_driver（见 [1.3 livox_ros_driver](#1.3)）**
- 若使用自定义 PCL，在 ~/.bashrc 中添加：`export PCL_ROOT={CUSTOM_PCL_PATH}`

## 3. 直接运行

注意：

A. 请确保 IMU 与 LiDAR **已同步**，这很重要。

B. 出现 “Failed to find match for field 'time'.” 表示 rosbag 中缺少每个 LiDAR 点的时间戳，会影响前向与反向传播。

C. 若已提供外参，建议将 **extrinsic_est_en** 设为 false。外参初始化可参考 [**Robust Real-time LiDAR-inertial Initialization**](https://github.com/hku-mars/LiDAR_IMU_Init)。

### 3.1 使用 ros launch 运行

按 [Livox-ros-driver2 安装](https://github.com/Livox-SDK/livox_ros_driver2) 将 PC 与 Livox LiDAR 连接后：

```bash
cd <ros2_ws>
. install/setup.bash  # 使用 zsh 则用 setup.zsh
ros2 launch fast_lio mapping.launch.py config_file:=avia.yaml
```

根据需要将 `config_file` 改为 config 目录下其他 yaml（如 **airy.yaml** 用于 Airy 雷达）。

启动 livox ros driver，以 MID360 为例：

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

- 对 Livox 系列，FAST-LIO 仅支持由 `livox_lidar_msg.launch` 采集的数据，因其 `livox_ros_driver2/CustomMsg` 提供每个 LiDAR 点的时间戳，对运动畸变校正很重要。`livox_lidar.launch` 目前无法提供。
- 修改帧率请先修改 [Livox-ros-driver](https://github.com/Livox-SDK/livox_ros_driver2) 中 [livox_lidar_msg.launch](https://github.com/Livox-SDK/livox_ros_driver/blob/master/livox_ros_driver2/launch/livox_lidar_msg.launch) 的 **publish_freq** 再编译 livox_ros_driver。

### 3.2 带外置 IMU 的 Livox 系列

mapping_avia.launch 理论上支持 mid-70、mid-40 等 Livox 系列 LiDAR，运行前需配置：

编辑 `config/avia.yaml`（或 `config/airy.yaml` 等）设置：

1. LiDAR 点云话题：`lid_topic`
2. IMU 话题：`imu_topic`
3. 平移外参：`extrinsic_T`
4. 旋转外参：`extrinsic_R`（仅支持旋转矩阵）
- FAST-LIO 中外参定义为 LiDAR 在 IMU 机体坐标系下的位姿（IMU 为基准）。可在官方手册中查找。
- FAST-LIO 提供简单的 Livox 软件时间同步，将 `time_sync_en` 设为 true 开启。仅在无法做外部时间同步时使用，软件同步无法保证精度。

### 3.4 PCD 保存

在 launch 文件中将 `pcd_save_enable` 设为 `1`。FAST-LIO 退出后，所有扫描（全局坐标系）将累积并保存到 `FAST_LIO/PCD/scans.pcd`。可用 `pcl_viewer scans.pcd` 查看点云。

*pcl_viewer 快捷键：*
- 运行中按 1、2、3、4、5 切换显示/着色方式：
```
    1 随机
    2 X 值
    3 Y 值
    4 Z 值
    5 强度
```

## 4. Rosbag 示例

### 4.1 Livox Avia Rosbag

<div align="left">
<img src="doc/results/HKU_LG_Indoor.png" width=47% />
<img src="doc/results/HKU_MB_002.png" width = 51% >
</div>

文件可从 [Google Drive](https://drive.google.com/drive/folders/1CGYEJ9-wWjr8INyan6q1BZz_5VtGB-fP?usp=sharing) 下载。**!!!需将 ros1 bag 转为 ros2!!!**

运行：

```bash
ros2 launch fast_lio mapping.launch.py config_path:=<你的配置文件路径>
ros2 bag play <你的 bag 目录>
```

### 4.2 Velodyne HDL-32E Rosbag

**NCLT 数据集**：原始 bin 见 [这里](http://robots.engin.umich.edu/nclt/)。

我们提供了 [Rosbag 文件](https://drive.google.com/drive/folders/1VBK5idI1oyW0GC_I_Hxh63aqam3nocNK?usp=sharing) 和 [Python 脚本](https://drive.google.com/file/d/1leh7DxbHx29DyS1NJkvEfeNJoccxH7XM/view) 生成 Rosbag：`python3 sensordata_to_rosbag_fastlio.py bin_file_dir bag_name.bag`。**!!!需将 ros1 bag 转为 ros2!!!** 转换方法见 [Convert rosbag versions](https://ternaris.gitlab.io/rosbags/topics/convert.html)。

运行：

```
roslaunch fast_lio mapping_velodyne.launch
rosbag play YOUR_DOWNLOADED.bag
```

## 5. 在 UAV 上的实现

为验证 FAST-LIO 在实际移动机器人上的鲁棒性与计算效率，我们搭建了小型四旋翼，搭载 70° 视场 Livox Avia LiDAR 与 DJI Manifold 2-C 机载电脑（1.8 GHz Intel i7-8550U，8G RAM），如下图所示。

机体主要结构为 3D 打印（铝或 PLA），.stl 文件将后续开源。

<div align="center">
    <img src="doc/uav01.jpg" width=40.5% >
    <img src="doc/uav_system.png" width=57% >
</div>

## 6. 致谢

感谢 LOAM（J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time）、[Livox_Mapping](https://github.com/Livox-SDK/livox_mapping)、[LINS](https://github.com/ChaoqinRobotics/LINS---LiDAR-inertial-SLAM) 与 [Loam_Livox](https://github.com/hku-mars/loam_livox)。
