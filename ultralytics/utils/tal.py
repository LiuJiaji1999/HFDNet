# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from .checks import check_version
from .metrics import bbox_iou, probiou, wasserstein_loss
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")

# shape(bs, n_max_labels, h*w)
# n_max_labels: 一个batch中一张图片中的gt的数量(一个batch中所有图片的gt的数量进行比较, 选出gt数量最大的那个作为n_max_labels)
# h*w = 80*80 + 40*40 + 20 * 20: 既是锚点的数量也是预测框的数量

class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk # # 每个gt box最多选择topk个候选框作为正样本
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2) 这里的anc_points已经是映射到原始图片上的坐标中心点了
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        # batch_size 大小
        self.bs = pd_scores.shape[0]
        # 每个图片真实框个数不同，按照 真实框最大个数 进行补0对齐
        self.n_max_boxes = gt_bboxes.shape[1]

        # 没有真实框，直接返回结果
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )
        
        # 真实框的 mask，正负样本的匹配程度，iou值
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        # 对一个正样本匹配多个真实框的情况 进行调整
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        # # 设定一个动态权重，更加关注那些与真实目标对齐良好且重叠程度较高的预测框
        #  # 这个动态权重由overlaps和align_metric决定，那些align_metric和overlaps小的预测框，它们对应的target_scores也小
        # 这样求解bce_loss(分类损失)的时候相对来说会变小，这样模型在训练过程中就不会过多关注那些align_metric和overlaps小的预测框了
        # 注意align_metric是跟预测分类分数和ciou都有关，这样模型在预测的时候，分类分数和iou会保持一致性，尽可能的不会存在分类分数低，而iou高的情况。
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        # 筛选锚点在真实框内的预测框
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        
        # 预测框与真实框的匹配程度、iou值
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
       
        # 为了每张图片的真实框对齐，进行补0操作，mask_gt 用于确定有效真实框
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        
        # # 选择有效真实框, 锚点落在真实框内部, 该锚点对应的预测框与真实框的匹配度是topk
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def power_transform(self, array, power=2):
        return torch.where(array < 0.5, array ** power, array ** (1/power))
    
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, power=True):
        """
        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, na)
        return:
            align_metric (Tensor): shape(bs, max_num_obj, na)  
            返回匹配度, max_num_obj可以理解为gt, na可以理解为pd, 也就是将gt中的每一个都与na中的进行计算匹配度
            overlaps (Tensor): shape(bs, max_num_obj, na)  返回计算公式中的ciou
        """
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        # ind[0]的值为[[0,...,0], ..., [b, ..., b]]  shape(b, max_num_obj)
        # ind[1]的值为gt_labels  shape(b, max_num_obj)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
       
        # Get the scores of each grid for each gt cls
        # 构建一个shape为[self.bs, self.n_max_boxes, na]的全0的bbox_scores, 
        # # pd_scores  shape(b, na, 2) -> pd_scores[ind[0], :, ind[1]]: shape(b, max_num_obj, na)
        # pd_scores[ind[0], :, ind[1]]进行广播机制 ind[0]中的[0, 0], ind[1]中的[0, 0] 得到pd_scores[0, :, 0] 以此进行广播
        # 将pd_scores的预测分类分数赋值到对应的bbox_scores中(只赋值mask_in_gt中为1的位置)
    
        # pd_scores[ind[0]] 将每个batch的生成的预测框的重复 max_num_obj 次 size 大小变为 b*max_num_obj*num_total_anchors*num_classes
        # bbox_scores 的 size 为 b * max_num_obj * num_total_anchors，ind[1] 对类别进行得分进行选取
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w 分类分支的输出

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        # bbox_iou 的计算结果 的 size 为 b * max_num_obj * num_total_anchors * 1，所以进行维度的压缩
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes) # iou的计算
        if power:
            overlaps[mask_gt] = self.power_transform(overlaps[mask_gt].to(dtype=torch.float)).to(overlaps.dtype)
        
        # 预测框与真实框的匹配程度： 预测类别分值 ** alpha * 预测框与真实框的iou值** beta 
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Iou calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0) # 对小于0值的截断
        # return wasserstein_loss(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        # 第一个值为排序的数组，第二个值为该数组中获取到的元素在原数组中的位置标号
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        # 如果没有给出有效真实框的mask，通过真实框和预测框的匹配程度确定真实框的有效性
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        # 这三行是一体的, 因为gt_labels被展开了, bs * n_max_boxes
        # 所以要进行第二行代, 由于batch_ind是0~(bs-1)之间, target_gt_idx在0~(n_max_boxes-1), 因此处理后的代码target_gt_idx是在0~(n_max_boxes-1 + (bs-1)*n_max_boxes)之间
        # 第三行代码是一种广播机制, 假设target_gt_idx[1][20]=30(30这个值一定在(1*n_max_boxes)~(1*n_max_boxes+n_max_boxes-1))
        # 也就是target_labels[1][20]=gt_labels[30], target_labels中的值相当于在第一张图片第20个锚点处对应的是第一张图片第(30-n_max_boxes)的label值
        # 假设target_gt_idx[0][1] = 0, 这个0是mask_pos[0, :, 1]中的最大值为0, 也就代表pd1这一个anchor并没有匹配到gt,是负样本, 但是gt_labels[0]确是第一张图片的第一个gt_box的label值, 所以在下方需要将target_score中的负样本进行过滤(置0)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        # 过滤负样本, 负样本的位置的target_scores都为0, 只保留正样本的
        # target_bboxes的在生成box损失的会过滤
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 4)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        # # 计算gt_bboxes的左上角坐标(lt)和右下角坐标(rb)。将gt_bboxes重塑为(b*n_boxes, 1, 4), 然后使用chunk(2, 2)将其沿第2维(通道维度)分割成两部分。
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        # 计算每个anchor中心相对于每个 gt 的偏移量。首先, 将xy_centers添加一个新的维度(维度大小为1)，得到形状为(1, h*w, 4)的张量。
        # 然后, 分别计算anchor中心与每个 gt 左上角和右下角坐标的差值, 
        # 并将这两个差值连接在一起，得到形状为(bs, n_boxes, n_anchors, 4)的张量bbox_deltas。
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # 对于每个anchor中心和每个ground truth bounding box，计算它们之间的最小距离(在x轴和y轴上)
        # 这可以通过对bbox_deltas沿第3维(anchor中心维度)求最小值来实现, 结果是一个形状为(bs, n_boxes, h*w)的张量。
        # 判断这些最小距离是否大于一个很小的阈值eps(默认为1e-9)。如果大于eps，则认为该anchor中心与对应的ground truth bounding box有重叠。
        # 返回一个形状为(bs, n_boxes, h*w)的张量, 其中值为1表示对应的anchor中心与ground truth bounding box有重叠，值为0表示没有重叠。
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.

        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
            overlaps (Tensor): shape(b, n_max_boxes, h*w)

        Returns:
            target_gt_idx (Tensor): shape(b, h*w)
            fg_mask (Tensor): shape(b, h*w)
            mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        # 一个预测框匹配真实框的个数
        # 预测输出总共会有h*w个预测框, n_max_boxes对应的是gt, 如果这一维度存在sum求和大于1的情况
        # h*w=8400, 假设[b][0] > 1, 也就是[0]处的预测框同时被分给多个gt 
        fg_mask = mask_pos.sum(-2)
        # 一个预测框匹配真实框的个数 > 1 
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            # 一个预测框匹配真实框的位置
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            # 与预测框iou最高的 真实框的索引
            #  # overlaps就是CIoU  选择gt与pd ciou最大的那个位置索引  这个索引的值的维度是1, 值也就是在0-n_max_boxes-1之间
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            # one-hot编码
             # is_max_overlaps: [b, n_max_boxes, h*w], 中将is_max_overlaps中对应的n_max_boxes的维度赋值为1
            # 这个跟select_topk_candidates中的运用有异曲同工之妙
            # 最终的目的就是筛选出gt与pd中CIoU最大的那一维, 将pd对应的多个gt中CIoU最大的那个赋值为1, 其余赋值为0
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            #  正样本的mask
            fg_mask = mask_pos.sum(-2)
        # 正样本与之匹配的真实框的 索引
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, n_boxes, 5)

        Returns:
            (Tensor): shape(b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    '''
            各特征图每个像素点一个锚点即Anchors,即每个像素点只预测一个box
            故共有 80x80 + 40x40 + 20x20 = 8400个anchors
        '''
    # anc_points : 8400 x 2 ，每个像素中心点坐标
    # strides_tensor: 8400 x 1 ，每个像素的缩放倍数
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
