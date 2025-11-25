import torch
import numpy as np
from cuda_config import devices


def compute_color_based_prior(colors):
    """基于颜色的先验：鲜艳/显著颜色更可能重要"""
    # 计算颜色饱和度
    max_vals, _ = torch.max(colors, dim=1, keepdim=True)
    min_vals, _ = torch.min(colors, dim=1, keepdim=True)
    saturation = (max_vals - min_vals) / (max_vals + 1e-8)

    # 计算颜色亮度
    brightness = torch.mean(colors, dim=1, keepdim=True)

    # 计算颜色对比度（与平均颜色的差异）
    mean_color = torch.mean(colors, dim=0, keepdim=True)
    color_contrast = torch.norm(colors - mean_color, dim=1, keepdim=True)
    color_contrast = color_contrast / (torch.max(color_contrast) + 1e-8)  # 归一化

    # 组合颜色特征
    color_prior = 0.4 * saturation + 0.4 * brightness + 0.2 * color_contrast

    # 映射到合理范围 [0.5, 1.0]
    color_prior = 0.5 + 0.5 * torch.sigmoid(color_prior * 3 - 1)
    # [N, 1]
    return color_prior


def compute_scale_based_prior(scales):
    """基于尺度的先验：适中尺度更可能重要"""
    scale_norms = torch.norm(scales, dim=1)  # [N] 计算每个高斯球的总体尺度
    # 计算尺度的统计量
    mean_dist = torch.mean(scale_norms)
    std_dist = torch.std(scale_norms)

    # 理想尺度范围（基于统计）
    ideal_min = mean_dist - 0.5 * std_dist
    ideal_max = mean_dist + 0.5 * std_dist

    # 计算每个点的尺度适宜度
    scale_scores = torch.zeros_like(scale_norms)

    # 太小的高斯球
    too_small = scale_norms < ideal_min
    scale_scores[too_small] = scale_norms[too_small] / ideal_min

    # 太大的高斯球
    too_large = scale_norms > ideal_max
    scale_scores[too_large] = ideal_max / scale_norms[too_large]

    # 理想范围内的高斯球
    ideal_range = ~too_small & ~too_large
    if ideal_range.any():
        ideal_center = (ideal_min + ideal_max) / 2
        ideal_scores = 1.0 - torch.abs(scale_norms[ideal_range] - ideal_center) / ((ideal_max - ideal_min) / 2)
        scale_scores[ideal_range] = torch.clamp(ideal_scores, 0.1, 1.0)

    # 确保所有点都有合理的分数
    scale_scores = torch.clamp(scale_scores, 0.1, 1.0)
    scale_prior = 0.5 + 0.5 * torch.sigmoid(scale_scores * 3 - 1)
    # [N, 1]
    return scale_prior.unsqueeze(1)  # 添加维度以匹配其他先验

def combine_priors(color_prior, scale_prior, geometry_prior=None):
    """组合多个先验分量得到最终生存概率"""
    combined_prior = (
            0.7 * color_prior +  # 颜色重要性占70%
            0.3 * scale_prior  # 尺度适宜性占30%
    )
    # 最终调整，确保在合理范围内 [0.5, 1]
    final_probs = 0.5 + 0.5 * torch.sigmoid(combined_prior * 2 - 1)

    return final_probs


def likelihood(image, viewpoint_cam, viewspace_point_tensor, survival_prob, visibility_filter):
    # with torch.no_grad():
    # 1. 基于渲染误差的似然贡献
    # 计算每个像素的渲染误差
    pixel_error = torch.abs(image - viewpoint_cam.original_image.to(device=devices.train_device()))
    # 计算平均像素误差作为整体质量指标
    avg_pixel_error = pixel_error.mean()

    # 2. 基于可见性的似然贡献
    visibility_mask = torch.zeros_like(survival_prob, dtype=bool)
    visibility_mask[visibility_filter] = True
    visibility_contribution = torch.zeros_like(survival_prob)
    visibility_contribution[visibility_mask] = 1.0

    # 3. 基于梯度的似然贡献（仅对可见点）
    grad_contribution = torch.zeros_like(survival_prob)
    if visibility_filter.numel() > 0 and viewspace_point_tensor.grad is not None:
        # 获取可见点的梯度范数
        xyz_grad = viewspace_point_tensor.grad[visibility_filter].norm(dim=1)
        # 归一化梯度贡献
        if xyz_grad.numel() > 0:
            grad_contribution[visibility_filter] = torch.clamp(xyz_grad / (xyz_grad.max() + 1e-8), 0, 1)

    # 4. 组合似然函数
    # 高质量渲染（低误差）应增加生存概率
    error_factor = torch.exp(-avg_pixel_error * 15.0)  # 指数衰减，误差越小，因子越大

    # 可见性权重（可见点应有更高的生存概率）
    visibility_weight = 0.6

    # 梯度权重（梯度大的点可能需要更细致的表示，应保留）
    gradient_weight = 0.3

    # 基础权重
    base_weight = 0.1

    # 计算新的生存概率
    new_survival_prob = torch.zeros_like(survival_prob)

    # 对可见点
    visible_idx = visibility_mask
    new_survival_prob[visible_idx] = (visibility_weight * visibility_contribution[visible_idx] +
                                      gradient_weight * grad_contribution[visible_idx]
                                      + base_weight) * error_factor

    # 对不可见点，降低其生存概率，但设置最小值避免过早剪枝
    invisible_idx = ~visibility_mask
    # 不可见点的新生存概率 = 当前概率 * 衰减系数
    decay_factor = 0.95  # 衰减系数
    min_prob = 0.05  # 最小生存概率
    new_survival_prob[invisible_idx] = torch.max(
        survival_prob[invisible_idx] * decay_factor,
        torch.full_like(survival_prob[invisible_idx], min_prob)
    )

    # 确保概率在有效范围内 [0, 1]
    new_survival_prob = torch.clamp(new_survival_prob, 0.01, 0.99)

    torch.cuda.empty_cache()
    return new_survival_prob



def likelihood_with_prior_update(image, viewpoint_cam, viewspace_point_tensor, initial_survival_prob, 
                                visibility_filter, gaussians, iteration=None):
    """
    在初始生存概率基础上更新似然函数
    
    参数:
    - image: 渲染图像
    - viewpoint_cam: 视角相机
    - viewspace_point_tensor: 视图空间中的点张量
    - initial_survival_prob: 初始生存概率（作为先验）
    - visibility_filter: 可见性过滤器
    - gaussians: 高斯模型
    - iteration: 当前迭代次数（可选，用于动态调整更新策略）
    """
    with torch.no_grad():
        # 1. 计算像素级渲染误差
        gt_image = viewpoint_cam.original_image.to(device=devices.train_device())
        pixel_error = torch.abs(image - gt_image)
        avg_pixel_error = pixel_error.mean()
        
        # 2. 创建可见性掩码
        visibility_mask = torch.zeros_like(initial_survival_prob, dtype=bool)
        visibility_mask[visibility_filter] = True
        
        # 3. 计算梯度贡献（针对可见点）
        grad_contribution = torch.zeros_like(initial_survival_prob)
        if visibility_filter.numel() > 0 and viewspace_point_tensor.grad is not None:
            # 获取可见点的梯度范数
            xyz_grad = viewspace_point_tensor.grad[visibility_filter].norm(dim=1)
            # 归一化梯度贡献
            if xyz_grad.numel() > 0:
                grad_contribution[visibility_filter] = torch.clamp(xyz_grad / (xyz_grad.max() + 1e-8), 0, 1)
        
        # 4. 计算颜色一致性贡献（基于渲染颜色与GT颜色的匹配度）
        color_consistency = torch.zeros_like(initial_survival_prob)
        if visibility_filter.numel() > 0:
            # 对于可见点，计算颜色相似度
            color_similarity = 1.0 - (pixel_error.mean(dim=0) / 2.0)  # 转换为相似度 [0, 1]
            color_consistency[visibility_filter] = color_similarity.mean()
        
        # 5. 动态调整权重（根据迭代次数）
        if iteration is not None:
            # 早期迭代更依赖先验，后期更依赖实际渲染表现
            prior_weight = max(0.3, 1.0 - iteration / 30000.0)  # 从0.7逐渐降到0.3
        else:
            prior_weight = 0.5
        
        # 计算其他因素的权重
        likelihood_weight = 1.0 - prior_weight
        
        # 6. 计算似然更新部分
        # 渲染质量因子
        quality_factor = torch.exp(-avg_pixel_error * 15.0)  # 更陡峭的指数衰减
        
        # 对可见点的更新
        visible_update = (0.4 * visibility_mask.float() +  # 可见性贡献
                         0.4 * grad_contribution +           # 梯度贡献
                         0.2 * color_consistency) * quality_factor  # 颜色一致性贡献
        
        # 7. 结合先验和似然更新
        # 可见点：结合先验和当前似然
        new_survival_prob = torch.zeros_like(initial_survival_prob)
        visible_idx = visibility_mask
        new_survival_prob[visible_idx] = prior_weight * initial_survival_prob[visible_idx] + \
                                         likelihood_weight * visible_update[visible_idx]
        
        # 不可见点：基于先验衰减，但保留一定概率
        invisible_idx = ~visibility_mask
        # 根据迭代动态调整衰减因子
        decay_factor = 0.95 if iteration is None or iteration < 10000 else 0.9
        min_prob = 0.02
        new_survival_prob[invisible_idx] = torch.max(
            initial_survival_prob[invisible_idx] * decay_factor,
            torch.full_like(new_survival_prob[invisible_idx], min_prob)
        )
        
        # 8. 应用平滑过渡 - 避免剧烈变化
        survival_prob_diff = new_survival_prob - initial_survival_prob
        smoothed_diff = 0.3 * survival_prob_diff  # 限制单次更新幅度
        smoothed_survival_prob = initial_survival_prob + smoothed_diff
        
        # 9. 确保概率在有效范围内
        smoothed_survival_prob = torch.clamp(smoothed_survival_prob, 0.01, 0.99)
        
        # 10. 转换回logit空间并更新
        new_survival_logit = gaussians.inverse_survival_prob_activation(smoothed_survival_prob)
        gaussians._survival_logit = new_survival_logit

# def compute_approximate_density(points, k=5):
#     """计算近似点云密度（简化实现）"""
#     n_points = points.shape[0]
#     density_scores = torch.ones((n_points, 1), device=points.device)
#
#     # 如果点云太大，使用随机采样来估计密度
#     if n_points > 10000:
#         # 随机选择子集进行计算
#         sample_indices = torch.randperm(n_points)[:10000]
#         sample_points = points[sample_indices]
#
#         # 计算子集内的最近邻距离
#
#         points_np = sample_points.cpu().numpy()
#         nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(points_np))  # k+1因为包含自身
#         nbrs.fit(points_np)
#         distances, _ = nbrs.kneighbors(points_np)
#
#         # 平均距离（排除自身）
#         mean_distances = np.mean(distances[:, 1:], axis=1)  # 跳过自身
#         density = 1.0 / (mean_distances + 1e-8)
#
#         # 归一化密度分数
#         density_normalized = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-8)
#         density_scores_sample = torch.tensor(density_normalized, device=points.device).unsqueeze(1)
#
#         # 为所有点分配平均密度（简化处理）
#         mean_density = torch.mean(density_scores_sample)
#         density_scores = density_scores * mean_density
#         else:
#         # 小点云直接计算
#         from sklearn.neighbors import NearestNeighbors
#         points_np = points.cpu().numpy()
#         nbrs = NearestNeighbors(n_neighbors=min(k + 1, n_points))
#         nbrs.fit(points_np)
#         distances, _ = nbrs.kneighbors(points_np)
#
#         mean_distances = np.mean(distances[:, 1:], axis=1)  # 跳过自身
#         density = 1.0 / (mean_distances + 1e-8)
#         density_normalized = (density - np.min(density)) / (np.max(density) - np.min(density) + 1e-8)
#         density_scores = torch.tensor(density_normalized, device=points.device).unsqueeze(1)
#
#     return density_scores