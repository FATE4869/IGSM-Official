import torch


def noise_estimation_loss(model,
                          x: torch.Tensor,
                          t: torch.LongTensor, e: torch.Tensor, keepdim=False):
    # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float()).sample
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def noise_estimation_kd_loss(model, teacher,
                             x: torch.Tensor, t: torch.LongTensor, e: torch.Tensor, keepdim=False,
                             noise_weight=1.0, kd_weight=1.0):

    output = model(x, t.float()).sample
    with torch.no_grad():
        teacher_output = teacher(x, t.float()).sample
    mse_loss = (e - output).square().sum(dim=(1, 2, 3))
    kd_loss = (teacher_output - output).square().sum(dim=(1, 2, 3))
    if keepdim:
        return noise_weight * mse_loss + kd_weight * kd_loss , mse_loss, kd_loss
    else:
        mse_loss = mse_loss.mean(dim=0)
        kd_loss = kd_loss.mean(dim=0)
        return noise_weight * mse_loss + kd_weight * kd_loss, mse_loss, kd_loss
    # if keepdim:
    #     return kd_weight*(teacher_output - output).square().sum(dim=(1, 2, 3)) + noise_weight * (e - output).square().sum(dim=(1, 2, 3))
    #     # return 0.7*(teacher_output - output).square().sum(dim=(1, 2, 3)) + 0.3 * (e - output).square().sum(dim=(1, 2, 3))
    # else:
    #     return 0.7*(teacher_output - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + 0.3 * (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def noise_estimation_kd_jac_loss(model, teacher,
                                 x: torch.Tensor, t: torch.LongTensor, e: torch.Tensor, keepdim=False,
                                 noise_weight=1.0, kd_weight=1.0, jm_weight=1.0, k=1):
    x.requires_grad = True

    batch_size, c, h, w = x.shape
    vectors = torch.nn.functional.normalize(torch.randn(batch_size, c*h*w, k, device=x.device))

    y = model(x, t.float()).sample.view(batch_size, -1)
    mse_loss = (e.view(batch_size, -1) - y).square().sum(dim=(-1))

    yd = teacher(x, t.float()).sample.view(batch_size, -1)
    kd_loss = (yd - y).square().sum(dim=(-1))

    y_v = torch.sum(y.unsqueeze(-1) * vectors)
    yd_v = torch.sum(yd.unsqueeze(-1) * vectors)

    J_v = torch.autograd.grad(y_v, x, create_graph=True)[0].view(batch_size, c*h*w, -1)
    Jd_v = torch.autograd.grad(yd_v, x, create_graph=True)[0].view(batch_size, c*h*w, -1).detach()
    jac_loss = (Jd_v.square().sum((1, 2)) - J_v.square().sum((1, 2))).square()
    # import pdb
    # pdb.set_trace()
    if keepdim:
        return kd_weight * kd_loss + noise_weight * mse_loss + jm_weight * jac_loss, mse_loss, kd_loss, jac_loss
    else:
        mse_loss = mse_loss.mean(dim=0)
        kd_loss = kd_loss.mean(dim=0)
        jac_loss = jac_loss.mean(dim=0)
        return noise_weight * mse_loss + kd_weight * kd_loss + jm_weight * jac_loss, mse_loss, kd_loss, jac_loss


loss_registry = {
    'simple': noise_estimation_loss,
}
