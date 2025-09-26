import torch

@torch.no_grad()
def _logloss_T(logits: torch.Tensor, targets: torch.Tensor, T: float) -> float:
    """Удобная проверка финального качества при известной T."""
    z = (logits.to(torch.float64) / T)
    lse = torch.logsumexp(z, dim=1)                             # (N,)
    zy  = z.gather(1, targets.view(-1,1)).squeeze(1)            # z_{n,y_n}
    return float((lse - zy).mean().item())

def fit_temperature_lbfgs(
    logits: torch.Tensor,        # (N, C) вал. логиты z_{n,c}
    targets: torch.Tensor,       # (N,) целевые индексы классов y_n (int64)
    init_logT: float = 0.0,      # старт: T=1
    max_iter: int = 100
):
    """
    Возвращает:
      T_star: float — оптимальная температура
      val_loss: float — LogLoss на валидации при T_star
    """
    assert logits.ndim == 2 and targets.ndim == 1
    assert logits.size(0) == targets.size(0)
    device = torch.device("cpu")  # стабильно и достаточно быстро
    z = logits.detach().to(torch.float64).to(device)
    y = targets.detach().to(torch.int64).to(device)

    # параметризуем T = exp(alpha), чтобы T>0 автоматически
    alpha = torch.tensor(init_logT, dtype=torch.float64, requires_grad=True, device=device)
    optim = torch.optim.LBFGS([alpha], lr=1.0, max_iter=max_iter, line_search_fn=None)

    def closure():
        optim.zero_grad()
        T = torch.exp(alpha)                       # скаляр
        zT = z / T                                 # (N, C)
        lse = torch.logsumexp(zT, dim=1)           # (N,)
        zy  = zT.gather(1, y.view(-1,1)).squeeze(1)
        loss = (lse - zy).mean()                   # скаляр
        loss.backward()                            # ∂loss/∂alpha
        return loss

    with torch.enable_grad():
        loss_final = optim.step(closure)

    T_star = float(torch.exp(alpha).item())

    # # На всякий случай: если оптимум "улетел", мягко ограничим и пересчитаем loss
    # if not (0.01 <= T_star <= 100.0):
    #     T_star = min(max(T_star, 0.01), 100.0)

    val_loss = _logloss_T(logits, targets, T_star)
    return T_star, val_loss

# --- пример использования ---
# logits_val: torch.Tensor (N, C)
# y_val:      torch.Tensor (N,)
# T_star, val_loss = fit_temperature_lbfgs(logits_val, y_val)
# print(f"T*: {T_star:.4f}, Val LogLoss: {val_loss:.6f}")
# Затем на тесте делайте: logits_test_calibrated = logits_test / T_star
