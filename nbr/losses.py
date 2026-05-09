import torch
import torch.nn.functional as F


def partition_targets(
    targets: torch.Tensor,
    alpha_idf: torch.Tensor,
    tau_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Equation 22: Partition ground-truth basket into core and fill subsets
    using a threshold on alpha_IDF.

        C*_{T+1} = { i in B_{T+1} : alpha_IDF_i > tau_alpha }
        F*_{T+1} = B_{T+1} \\ C*_{T+1}

    Args:
        targets:   (B, T, V) binary ground-truth basket labels
        alpha_idf: (V,)      per-item IDF-corrected importance scores
        tau_alpha: float     threshold separating core from fill items

    Returns:
        core_targets: (B, T, V) — targets restricted to core items
        fill_targets: (B, T, V) — targets restricted to fill items
    """
    # (V,) boolean mask broadcast over (B, T, V)
    is_core = (alpha_idf > tau_alpha).float()   # 1 for core items, 0 for fill
    is_fill = 1.0 - is_core                     # complement

    core_targets = targets * is_core
    fill_targets = targets * is_fill

    return core_targets, fill_targets


def intent_loss(
    logits: torch.Tensor,
    core_targets: torch.Tensor,
    alpha_idf: torch.Tensor,
) -> torch.Tensor:
    """
    Equation 23: Importance-weighted BCE over core items only.

    Missing a load-bearing item incurs a larger penalty (scaled by alpha_IDF).
    Negative targets (items not in basket) receive weight 1.0.

        L_intent = - (1/T) sum_t sum_j alpha_IDF_j [
                       y_{t,j} log sigma(s^intent_j)
                     + (1 - y_{t,j}) log(1 - sigma(s^intent_j))
                   ]

    The per-sample weighting is:
        weight_j = alpha_IDF_j   if y_{t,j} = 1  (positive: load-bearing item)
                 = 1.0            if y_{t,j} = 0  (negative: not in basket)

    Args:
        logits:       (B, T, V) raw intent scores s^intent
        core_targets: (B, T, V) binary labels restricted to core items
                                (fill item positions are 0, so they contribute
                                 only as negatives — their alpha_IDF weight on
                                 the positive term is zeroed out by the target)
        alpha_idf:    (V,)      per-item importance scores, normalised to [0,1]

    Returns:
        Scalar intent loss.
    """
    # Positive positions weighted by alpha_IDF; negative positions weighted 1.0
    # Shape: (V,) broadcasts correctly over (B, T, V)
    weights = core_targets * alpha_idf + (1.0 - core_targets)

    return F.binary_cross_entropy_with_logits(
        logits, core_targets, weight=weights
    )


def fill_loss(
    logits: torch.Tensor,
    fill_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Equation 24: Uniform-weight BCE over fill (peripheral) items only.

    The fill head is never asked to predict core items — fill_targets has
    zeros at all core item positions, so they contribute only as negatives.
    No alpha_IDF weighting: all positions are treated equally.

        L_fill = - (1/T) sum_t sum_{j in F*} [
                     y_{t,j} log sigma(s^fill|intent_j)
                   + (1 - y_{t,j}) log(1 - sigma(s^fill|intent_j))
                 ]

    Args:
        logits:       (B, T, V) raw fill scores s^fill|intent
        fill_targets: (B, T, V) binary labels restricted to fill items

    Returns:
        Scalar fill loss.
    """
    return F.binary_cross_entropy_with_logits(logits, fill_targets)


def mlm_loss(
    mlm_logits: torch.Tensor,
    mlm_targets: torch.Tensor,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """
    Equation 3: Auxiliary masked language modelling loss.

    Cross-entropy between predicted item distributions at masked positions
    and the true item ids. Non-masked positions are ignored via ignore_index.

        L_MLM = CE( h_masked_k, i_true_k )

    Args:
        mlm_logits:   (B, T, S, V) per-position item logits from mlm_head
        mlm_targets:  (B, T, S)    true item ids at masked positions;
                                   non-masked positions should be set to
                                   mask_token_id so they are ignored.
        mask_token_id: int         index used to flag non-masked positions
                                   (typically the padding/[MASK] id, default 0)

    Returns:
        Scalar MLM loss.
    """
    # Flatten spatial dims: (B*T*S, V) vs (B*T*S,)
    return F.cross_entropy(
        mlm_logits.view(-1, mlm_logits.size(-1)),
        mlm_targets.view(-1),
        ignore_index=mask_token_id,
    )


def orthogonality_loss(
    intent_repr: torch.Tensor,
    fill_repr: torch.Tensor,
) -> torch.Tensor:
    """
    Equation in Section F: Orthogonality regulariser.

    Penalises drift of the low-rank projection P toward the identity by
    keeping intent and fill components orthogonal:

        L_orth = || h^intent · h^fill ||^2

    By construction h^intent + h^fill = h_{T+1} exactly, but without this
    regulariser the learned P can collapse so that one component dominates.

    Args:
        intent_repr: (B, T, D)
        fill_repr:   (B, T, D)

    Returns:
        Scalar orthogonality loss.
    """
    # Dot product per sample, then mean over batch and time
    dot = (intent_repr * fill_repr).sum(dim=-1)   # (B, T)
    return (dot ** 2).mean()


def total_loss(
    intent_logits: torch.Tensor,
    fill_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_idf: torch.Tensor,
    tau_alpha: float,
    intent_repr: torch.Tensor,
    fill_repr: torch.Tensor,
    mlm_logits: torch.Tensor,
    mlm_targets: torch.Tensor,
    weights: dict,
    mask_token_id: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Equation 25: Full training objective.

        L = L_intent + lambda * L_fill + gamma * L_orth + eta * L_MLM

    Steps:
        1. Partition ground-truth targets into core / fill (Eq. 22)
        2. Compute L_intent on core partition (Eq. 23)
        3. Compute L_fill on fill partition (Eq. 24)
        4. Compute L_orth from the projected intent / fill representations
        5. Compute L_MLM from the intra-basket encoder auxiliary task (Eq. 3)
        6. Weighted sum with caller-supplied coefficients

    Args:
        intent_logits: (B, T, V)   raw scores from the intent head
        fill_logits:   (B, T, V)   raw scores from the fill head
        targets:       (B, T, V)   full binary ground-truth basket labels
        alpha_idf:     (V,)        per-item IDF importance scores
        tau_alpha:     float       threshold for core / fill partition
        intent_repr:   (B, T, D)   h^intent from the orthogonal projection
        fill_repr:     (B, T, D)   h^fill  from the orthogonal projection
        mlm_logits:    (B, T, S, V) per-position item logits from mlm_head
        mlm_targets:   (B, T, S)   true item ids at masked positions only
        weights:       dict with keys "intent", "fill", "orth", "mlm"
        mask_token_id: int         ignore_index for MLM cross-entropy

    Returns:
        (total_loss scalar, dict of individual loss values for logging)
    """
    # Step 1 — partition targets (Eq. 22)
    core_targets, fill_targets = partition_targets(targets, alpha_idf, tau_alpha)

    # Step 2 — importance-weighted intent BCE (Eq. 23)
    l_intent = intent_loss(intent_logits, core_targets, alpha_idf)

    # Step 3 — uniform fill BCE on peripheral items only (Eq. 24)
    l_fill = fill_loss(fill_logits, fill_targets)

    # Step 4 — orthogonality regulariser
    l_orth = orthogonality_loss(intent_repr, fill_repr)

    # Step 5 — auxiliary MLM loss (Eq. 3)
    if mlm_logits is not None and mlm_targets is not None:
        l_mlm = mlm_loss(mlm_logits, mlm_targets, mask_token_id)
    else:
        l_mlm = torch.tensor(0.0, device=intent_logits.device)


    # Step 6 — weighted sum (Eq. 25)
    L_total = (
        weights.get("intent", 1.0) * l_intent
        + weights.get("fill",   1.0) * l_fill
        + weights.get("orth",   0.1) * l_orth
        + weights.get("mlm",    0.5) * l_mlm
    )

    return L_total, {
        "loss/total":   L_total.item(),
        "loss/intent":  l_intent.item(),
        "loss/fill":    l_fill.item(),
        "loss/orth":    l_orth.item(),
        "loss/mlm":     l_mlm.item(),
    }