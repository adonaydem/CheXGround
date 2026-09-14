from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

SAFE_NORMALIZE_EPS = 1e-6


def _ensure_effective_roi_mask(roi_valid_mask: torch.BoolTensor) -> torch.BoolTensor:
    effective_mask = roi_valid_mask.clone()
    no_valid_roi = ~effective_mask.any(dim=1)
    if no_valid_roi.any():
        effective_mask[no_valid_roi, 0] = True
    return effective_mask


def _pack_phrases_by_image(
    phrase_embeddings: torch.Tensor,
    phrase_batch_index: torch.LongTensor,
    batch_size: int,
    extra_tensor: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    device = phrase_embeddings.device
    dtype = phrase_embeddings.dtype
    phrase_counts = torch.bincount(phrase_batch_index, minlength=batch_size)
    max_phrases = int(phrase_counts.max().item()) if phrase_counts.numel() > 0 else 0

    packed_phrases = phrase_embeddings.new_zeros((batch_size, max_phrases, phrase_embeddings.shape[-1]))
    phrase_valid_mask = torch.zeros((batch_size, max_phrases), dtype=torch.bool, device=device)
    packed = {
        "phrase_embeddings": packed_phrases,
        "phrase_valid_mask": phrase_valid_mask,
        "phrase_counts": phrase_counts,
    }
    if extra_tensor is not None:
        packed["extra_tensor"] = extra_tensor.new_zeros((batch_size, max_phrases, *extra_tensor.shape[1:]))

    if phrase_embeddings.shape[0] == 0 or max_phrases == 0:
        return packed

    sort_order = torch.argsort(phrase_batch_index)
    sorted_batch_index = phrase_batch_index.index_select(0, sort_order)
    batch_offsets = torch.cumsum(phrase_counts, dim=0) - phrase_counts
    local_positions = torch.arange(sort_order.shape[0], device=device, dtype=torch.long)
    local_positions = local_positions - batch_offsets.index_select(0, sorted_batch_index)

    packed_phrases[sorted_batch_index, local_positions] = phrase_embeddings.index_select(0, sort_order)
    phrase_valid_mask[sorted_batch_index, local_positions] = True
    if extra_tensor is not None:
        packed["extra_tensor"][sorted_batch_index, local_positions] = extra_tensor.index_select(0, sort_order)

    return packed


def image_to_text_contrastive_loss(
    pair_logits: torch.Tensor,
    valid_mask: torch.BoolTensor,
) -> torch.Tensor:
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return pair_logits.sum() * 0.0
    logits = pair_logits.index_select(0, valid_indices).index_select(1, valid_indices)
    targets = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


def gloria_positive_cosine_loss(
    region_embeddings: torch.Tensor,
    phrase_embeddings: torch.Tensor,
    phrase_batch_index: torch.LongTensor,
    phrase_positive_roi_mask: torch.BoolTensor,
    roi_valid_mask: torch.BoolTensor,
) -> torch.Tensor:
    if phrase_embeddings.numel() == 0 or phrase_positive_roi_mask.numel() == 0:
        return region_embeddings.sum() * 0.0

    aligned_region_embeddings = region_embeddings.index_select(0, phrase_batch_index)
    aligned_roi_valid_mask = roi_valid_mask.index_select(0, phrase_batch_index)
    positive_mask = phrase_positive_roi_mask & aligned_roi_valid_mask
    if not positive_mask.any():
        return region_embeddings.sum() * 0.0

    pairwise_similarity = torch.einsum(
        "pkd,pd->pk",
        aligned_region_embeddings,
        phrase_embeddings,
    )
    positive_loss = 1.0 - pairwise_similarity
    return positive_loss[positive_mask].mean()


def gloria_attention_kl_loss(
    region_embeddings: torch.Tensor,
    phrase_attention_embeddings: torch.Tensor,
    phrase_similarity_embeddings: torch.Tensor,
    phrase_positive_roi_mask: torch.BoolTensor,
    phrase_batch_index: torch.LongTensor,
    roi_valid_mask: torch.BoolTensor,
    soft_target_alpha: float = 0.8,
) -> torch.Tensor:
    if phrase_attention_embeddings.numel() == 0 or phrase_positive_roi_mask.numel() == 0:
        return region_embeddings.sum() * 0.0

    batch_size = region_embeddings.shape[0]
    packed_attention = _pack_phrases_by_image(
        phrase_embeddings=phrase_attention_embeddings,
        phrase_batch_index=phrase_batch_index,
        batch_size=batch_size,
        extra_tensor=phrase_positive_roi_mask,
    )
    packed_similarity = _pack_phrases_by_image(
        phrase_embeddings=phrase_similarity_embeddings,
        phrase_batch_index=phrase_batch_index,
        batch_size=batch_size,
    )
    packed_phrase_attention_embeddings = packed_attention["phrase_embeddings"]
    packed_phrase_similarity_embeddings = packed_similarity["phrase_embeddings"]
    phrase_valid_mask = packed_attention["phrase_valid_mask"]
    packed_phrase_positive_roi_mask = packed_attention["extra_tensor"]

    if packed_phrase_attention_embeddings.shape[1] == 0:
        return region_embeddings.sum() * 0.0

    report_valid_mask = phrase_valid_mask.any(dim=1)
    roi_to_phrase_logits = torch.einsum("bkd,bpd->bkp", region_embeddings, packed_phrase_attention_embeddings)
    roi_to_phrase_logits = roi_to_phrase_logits.masked_fill(
        ~phrase_valid_mask[:, None, :],
        torch.finfo(roi_to_phrase_logits.dtype).min,
    )
    if (~report_valid_mask).any():
        roi_to_phrase_logits = roi_to_phrase_logits.clone()
        roi_to_phrase_logits[~report_valid_mask] = 0
    roi_to_phrase_log_attention = F.log_softmax(roi_to_phrase_logits, dim=-1)

    hard_target = packed_phrase_positive_roi_mask.transpose(1, 2).to(dtype=roi_to_phrase_log_attention.dtype)
    supervised_mask = hard_target.any(dim=-1) & roi_valid_mask
    if not supervised_mask.any():
        return region_embeddings.sum() * 0.0

    hard_target = hard_target / hard_target.sum(dim=-1, keepdim=True).clamp(min=1.0)
    phrase_similarity = torch.matmul(
        packed_phrase_similarity_embeddings,
        packed_phrase_similarity_embeddings.transpose(1, 2),
    )
    phrase_similarity = phrase_similarity.masked_fill(
        ~phrase_valid_mask[:, None, :],
        torch.finfo(phrase_similarity.dtype).min,
    )
    if (~report_valid_mask).any():
        phrase_similarity = phrase_similarity.clone()
        phrase_similarity[~report_valid_mask] = 0
    phrase_similarity_target = F.softmax(phrase_similarity, dim=-1).detach()

    # Aggregate soft targets from all phrases linked to that ROI, selected by hard positives.
    soft_target = torch.matmul(hard_target, phrase_similarity_target)

    fused_target = (1.0 - soft_target_alpha) * hard_target + soft_target_alpha * soft_target
    fused_target = fused_target / fused_target.sum(dim=-1, keepdim=True).clamp(min=1.0)

    positive_entries = (fused_target > 0) & supervised_mask[:, :, None]
    selected_target = fused_target[positive_entries]
    selected_log_attention = roi_to_phrase_log_attention[positive_entries]
    kl_terms = selected_target * (torch.log(selected_target.clamp(min=1e-8)) - selected_log_attention)
    supervised_roi_count = supervised_mask.sum().to(dtype=region_embeddings.dtype)
    return kl_terms.sum() / supervised_roi_count


class GLoRIAROILocalMatcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        region_embeddings: torch.Tensor,
        roi_valid_mask: torch.BoolTensor,
        phrase_attention_embeddings: torch.Tensor,
        phrase_score_embeddings: torch.Tensor,
        phrase_batch_index: torch.LongTensor,
        temp1: float,
        temp2: float,
        temp3: float,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_rois, _ = region_embeddings.shape
        num_phrases = phrase_attention_embeddings.shape[0]
        device = region_embeddings.device
        dtype = region_embeddings.dtype

        if num_phrases == 0:
            return {
                "pair_logits": region_embeddings.new_zeros((batch_size, batch_size)),
                "valid_mask": torch.zeros((batch_size,), dtype=torch.bool, device=device),
                "image_phrase_mask": torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
            }

        image_ids = torch.arange(batch_size, device=device, dtype=phrase_batch_index.dtype)
        image_phrase_mask = image_ids[:, None] == phrase_batch_index[None, :]
        report_valid_mask = image_phrase_mask.any(dim=1)
        effective_roi_mask = _ensure_effective_roi_mask(roi_valid_mask)
        local_valid_mask = report_valid_mask & roi_valid_mask.any(dim=1)
        packed = _pack_phrases_by_image(
            phrase_embeddings=phrase_attention_embeddings,
            phrase_batch_index=phrase_batch_index,
            batch_size=batch_size,
            extra_tensor=phrase_score_embeddings,
        )
        packed_phrase_attention_embeddings = packed["phrase_embeddings"]
        packed_phrase_score_embeddings = packed["extra_tensor"]
        phrase_valid_mask = packed["phrase_valid_mask"]
        phrase_counts = packed["phrase_counts"].clamp(min=1).to(dtype=dtype)

        pair_logits = torch.zeros((batch_size, batch_size), dtype=dtype, device=device)

        for image_index in range(batch_size):
            roi_mask = effective_roi_mask[image_index]
            roi_features = region_embeddings[image_index, roi_mask]
            attention_logits = torch.einsum("bpd,rd->bpr", packed_phrase_attention_embeddings, roi_features)
            attention_logits = attention_logits * temp1
            log_attention = F.log_softmax(attention_logits, dim=-1)
            attention = log_attention.exp()

            phrase_context = torch.einsum("bpr,rd->bpd", attention, roi_features)
            phrase_context = F.normalize(phrase_context, dim=-1, eps=SAFE_NORMALIZE_EPS)
            phrase_context = torch.nan_to_num(phrase_context, nan=0.0, posinf=0.0, neginf=0.0)
            phrase_scores = (phrase_context * packed_phrase_score_embeddings).sum(dim=-1)
            phrase_scores = phrase_scores.masked_fill(~phrase_valid_mask, torch.finfo(dtype).min)

            image_pair_logits = torch.logsumexp(phrase_scores * temp2, dim=-1)
            image_pair_logits = image_pair_logits - phrase_counts.log()
            image_pair_logits = torch.where(
                report_valid_mask,
                image_pair_logits,
                torch.zeros_like(image_pair_logits),
            )
            pair_logits[image_index] = image_pair_logits

        pair_logits = pair_logits * temp3
        pair_logits = torch.where(local_valid_mask[:, None], pair_logits, torch.zeros_like(pair_logits))
        pair_logits = torch.nan_to_num(pair_logits, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "pair_logits": pair_logits,
            "valid_mask": local_valid_mask,
            "image_phrase_mask": image_phrase_mask,
        }
