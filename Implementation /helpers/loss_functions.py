import torch


def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    bsum = lambda x: torch.sum(x, dim=1)
    def mSDRLoss(orig, est):
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)

def SNR_loss(target, estimate):

    loss = 10*torch.log10( (torch.norm(target-estimate, p=2, dim=1)**2)/(torch.norm(target, p=2, dim=1)**2+ 1e-8) )

    return torch.mean(loss)