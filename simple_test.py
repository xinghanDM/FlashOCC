import torch
import time
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel

import projects.mmdet3d_plugin.models

# ROS2 imports
try:
    import rclpy
    from ros2_visualizer import FlashOCCVisualizer
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available, skipping visualization")
    ROS2_AVAILABLE = False  


def main():
    # ----------------------------
    # 1. 初始化ROS2 (如果可用)
    # ----------------------------
    visualizer = None
    if ROS2_AVAILABLE:
        rclpy.init()
        visualizer = FlashOCCVisualizer()
        print("ROS2 visualizer initialized")

    # ----------------------------
    # 2. 读取配置文件
    # ----------------------------
    cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
    cfg.data.test.test_mode = True

    # ----------------------------
    # 3. 构建模型并加载 checkpoint
    # ----------------------------
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, "ckpts/flashocc-r50-256x704.pth", map_location="cpu")

    # 设置类别信息
    dataset = build_dataset(cfg.data.test)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # 移动模型到 GPU (使用MMDataParallel但优化内存)
    model = MMDataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    # 清理GPU缓存
    torch.cuda.empty_cache()

    # ----------------------------
    # 4. 构建 dataloader
    # ----------------------------
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,  # 设置为0以节省内存
        dist=False,
        shuffle=False
    )

    # ----------------------------
    # 5. 前向推理和可视化
    # ----------------------------
    total_batches = len(data_loader)
    print(f"开始处理数据集，总共 {total_batches} 个批次")
    
    for i, data in enumerate(data_loader):
        try:
            # 把 tensor 数据搬到 GPU (MMDataParallel会自动处理)
            data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data.items()}

            with torch.no_grad():
                start_time = time.time()
                result = model(return_loss=False, rescale=True, **data)
                end_time = time.time()
            
            print(f"Batch {i+1}/{total_batches} 推理时间：{end_time - start_time:.3f}秒")
            
            # ----------------------------
            # 6. ROS2可视化 (每10个batch发布一次，减少发布频率)
            # ----------------------------
            if visualizer is not None and isinstance(result, list) and len(result) > 0:
                # 结果直接是occupancy grid数组
                occ_grid = result[0]
                if isinstance(occ_grid, np.ndarray):
                    print(f"发布occupancy grid (Batch {i+1}): {occ_grid.shape}")
                    visualizer.publish_occupancy(occ_grid)
                    # 处理ROS2消息
                    rclpy.spin_once(visualizer, timeout_sec=0.1)
                else:
                    print(f"结果类型不是numpy数组: {type(occ_grid)}")
            else:
                print(f"Batch {i+1} 推理完成，无有效结果")
            
            # 清理GPU缓存和删除不需要的变量以释放内存
            torch.cuda.empty_cache()
            del data, result
                
        except Exception as e:
            print(f"处理Batch {i+1}时出错: {str(e)}")
            continue
            
    # ----------------------------
    # 7. 清理ROS2
    # ----------------------------
    if visualizer is not None:
        visualizer.destroy_node()
        rclpy.shutdown()
        print("ROS2 visualizer shutdown")

if __name__ == "__main__":
    main()
