#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct

# 定义occupancy类别名称和颜色映射 (基于实际出现的类别)
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

# 将color_map转换为numpy数组用于向量化操作
color_array = np.array([
    [128, 128, 128],  # 0: others
    [255, 0, 0],      # 1: barrier
    [0, 255, 0],      # 2: bicycle
    [0, 0, 255],      # 3: bus
    [255, 255, 0],    # 4: car
    [255, 0, 255],    # 5: construction_vehicle
    [0, 255, 255],    # 6: motorcycle
    [255, 128, 0],    # 7: pedestrian
    [128, 0, 255],    # 8: traffic_cone
    [255, 192, 203],  # 9: trailer
    [0, 128, 0],      # 10: truck
    [128, 128, 0],    # 11: driveable_surface
    [0, 128, 128],    # 12: other_flat
    [128, 0, 128],    # 13: sidewalk
    [192, 192, 192],  # 14: terrain
    [255, 165, 0],    # 15: manmade
    [0, 255, 0],      # 16: vegetation
    [0, 0, 0]         # 17: free
], dtype=np.uint8)

class FlashOCCVisualizer(Node):
    def __init__(self):
        super().__init__('flashocc_visualizer')
        self.publisher_ = self.create_publisher(PointCloud2, '/flashocc/occupancy', 10)
        self.get_logger().info('FlashOCC ROS2 Visualizer started')
        
    def publish_occupancy(self, occupancy_grid):
        """
        将occupancy grid转换为PointCloud2消息并发布
        
        Args:
            occupancy_grid: numpy array of shape (H, W, D) 或 (H, W, D, C)
        """
        try:
            # 确保输入是numpy数组
            if not isinstance(occupancy_grid, np.ndarray):
                occupancy_grid = np.array(occupancy_grid)
            
            # 获取网格维度
            if len(occupancy_grid.shape) == 3:
                H, W, D = occupancy_grid.shape
                C = 1  # 单通道
            elif len(occupancy_grid.shape) == 4:
                H, W, D, C = occupancy_grid.shape
            else:
                self.get_logger().error(f"不支持的occupancy grid形状: {occupancy_grid.shape}")
                return
            
            # 定义体素大小和范围 (基于grid_config_occ)
            voxel_size = 0.4  # 40cm per voxel
            x_range = [-40, 40]  # X范围：-40m到+40m
            y_range = [-40, 40]  # Y范围：-40m到+40m  
            z_range = [-1, 5.4]  # Z范围：-1m到+5.4m
            
            # 向量化实现：创建3D坐标网格
            h_indices, w_indices, d_indices = np.meshgrid(
                np.arange(H), np.arange(W), np.arange(D), indexing='ij'
            )
            
            # 计算所有体素的3D坐标
            x_coords = x_range[0] + w_indices * voxel_size
            y_coords = y_range[0] + h_indices * voxel_size  
            z_coords = z_range[0] + d_indices * voxel_size
            
            # 处理多通道情况
            if C == 1:
                # 单通道：直接使用occupancy_grid
                voxel_values = occupancy_grid
            else:
                # 多通道：找到每个体素的最大值通道
                voxel_values = np.argmax(occupancy_grid, axis=-1)
            
            # 创建掩码：显示所有非空体素，但排除类别17(free)
            mask = (voxel_values > 0) & (voxel_values != 17) & (voxel_values != 15)
            
            if not np.any(mask):
                return
            
            # 提取被占用体素的坐标和类别
            points = np.column_stack([
                x_coords[mask],
                y_coords[mask], 
                z_coords[mask]
            ]).astype(np.float32)
            
            # 向量化颜色映射
            class_ids = voxel_values[mask].astype(np.int32)
            # 确保class_id在有效范围内
            class_ids = np.clip(class_ids, 0, len(color_array) - 1)
            colors = color_array[class_ids]
            
            # 创建PointCloud2消息
            point_cloud_msg = PointCloud2()
            point_cloud_msg.header = Header()
            point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
            point_cloud_msg.header.frame_id = 'base_link'
            
            # 设置字段
            point_cloud_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]
            
            # 设置数据
            point_cloud_msg.height = 1
            point_cloud_msg.width = len(points)
            point_cloud_msg.point_step = 16  # 3*4 + 1*4 = 16 bytes
            point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
            point_cloud_msg.is_bigendian = False
            point_cloud_msg.is_dense = True
            
            # 打包数据
            data = []
            for i in range(len(points)):
                # 位置 (x, y, z)
                data.extend(struct.pack('fff', points[i][0], points[i][1], points[i][2]))
                # 颜色 (rgb packed as UINT32)
                rgb_packed = (int(colors[i][0]) << 16) | (int(colors[i][1]) << 8) | int(colors[i][2])
                data.extend(struct.pack('I', rgb_packed))
            
            point_cloud_msg.data = bytes(data)
            
            # 发布消息
            try:
                self.publisher_.publish(point_cloud_msg)
            except Exception as e:
                self.get_logger().error(f"发布PointCloud2消息失败: {str(e)}")
                raise
            
        except Exception as e:
            self.get_logger().error(f"发布occupancy grid时出错: {str(e)}")

def main():
    rclpy.init()
    node = FlashOCCVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
