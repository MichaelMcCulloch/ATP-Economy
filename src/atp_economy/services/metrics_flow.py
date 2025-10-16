# src/atp_economy/services/metrics_flow.py
import torch
from ..domain.state import WorldState


def value_added_production(state: WorldState, rate_RJ: torch.Tensor) -> torch.Tensor:
    """
    GDP (flow) from transformation activities as Value Added:
      VA_r = sum_j [ p_r•(outputs of j) - p_r•(intermediate inputs of j) ] * rate_{rj}
    where S[g,j] < 0 are inputs, > 0 are outputs.

    Args:
      rate_RJ: [R, J] realized reaction rates this step
    Returns:
      VA_R: [R] value added per region
    """
    S = state.S  # [G,J]
    p_RG = state.price.T  # [R,G]
    S_pos = torch.clamp(S, min=0.0)  # outputs
    S_neg = torch.clamp(-S, min=0.0)  # inputs

    # Revenue and intermediate cost per region j
    rev_RJ = p_RG @ S_pos  # [R,J]
    int_RJ = p_RG @ S_neg  # [R,J]

    VA_RJ = (rev_RJ - int_RJ) * torch.clamp(rate_RJ, min=0.0)
    return VA_RJ.sum(dim=1)  # [R]


def value_added_extraction(state: WorldState, q_RM: torch.Tensor) -> torch.Tensor:
    """
    Value added from extraction of M resource goods (no intermediate inputs tracked here).
    Args:
      q_RM: [R, M] extraction quantities by region and resource index
    Returns:
      VA_R: [R]
    """
    goods_idx = state.res_goods  # [M]
    p_RM = state.price.index_select(0, goods_idx).T  # [R, M]
    return (p_RM * torch.clamp(q_RM, min=0.0)).sum(dim=1)
