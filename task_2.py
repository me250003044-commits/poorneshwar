# audio_cgan_imageHP.py
# Conditional DCGAN for audio with image-GAN hyperparameters, local run.

import os, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# 0) Weight init (DCGAN-style)
# -----------------------------
def weights_init_dcgan(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias.data)
        except Exception:
            pass
    if isinstance(m, nn.BatchNorm2d):
        try:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.zeros_(m.bias.data)
        except Exception:
            pass

# ===============================================================================
# 1) DATASET
# ===============================================================================
class TrainAudioSpectrogramDataset(Dataset):
    """
    Expects: root_dir/<class>/*.wav
    Outputs: (log-mel [1,128,512], onehot label [C])
    """
    def __init__(self, root_dir, categories=None, max_frames=512, fraction=1.0, sample_rate=22050):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        if categories is None:
            categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.categories = categories
        self.class_to_idx = {c: i for i, c in enumerate(categories)}
        self.file_list = []
        for c in categories:
            cdir = os.path.join(root_dir, c)
            if not os.path.exists(cdir): continue
            files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(".wav")]
            k = int(len(files) * fraction)
            if k == 0: continue
            for p in random.sample(files, k):
                self.file_list.append((p, self.class_to_idx[c]))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No wav files found in {root_dir}.")

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=256, n_mels=128
        )

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        wav, sr = sf.read(path)
        wav = torch.tensor(wav).float()
        if wav.dim() > 1: wav = wav.mean(dim=1)
        wav = wav.unsqueeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        mel = self.mel(wav)             # [1, 128, T]
        logmel = torch.log1p(mel)       # non-negative log domain
        T = logmel.shape[-1]
        if T < self.max_frames:
            logmel = F.pad(logmel, (0, self.max_frames - T))
        else:
            logmel = logmel[:, :, :self.max_frames]

        y = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()
        return logmel, y

# ===============================================================================
# 2) MODELS — image-GAN hyperparameters and style
# ===============================================================================
# Generator: Dense(256*4*4) → Reshape(256,4,4)
#            3× ConvT stride2 (image-GAN style) + extra blocks to reach 128×512,
#            LeakyReLU(0.2) everywhere like the image GAN, final Tanh in [-1,1],
#            then scaled back to log domain for audio inversion.
class CGAN_Generator(nn.Module):
    def __init__(self, latent_dim, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        self.fc = nn.Linear(latent_dim + num_categories, 256 * 4 * 4)

        blocks = []
        def up(in_c, out_c):  # ConvTranspose2d block with LeakyReLU(0.2)
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # Start: (256,4,4)
        # Image GAN uses 3 up blocks to 32×32; for audio we need 128×512 (×32 height, ×128 width)
        # We'll add enough stride-2 ups to exceed, then adaptively pool to exact shape.
        self.up = nn.Sequential(
            up(256, 128),  # 4→8
            up(128, 128),  # 8→16
            up(128, 128),  # 16→32
            up(128, 64),   # 32→64
            up(64, 32),    # 64→128
            up(32, 16),    # 128→256
            up(16, 8),     # 256→512
        )
        self.to_map = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Tanh(),                # match image GAN output range
            nn.AdaptiveAvgPool2d(spec_shape)  # force [1,128,512]
        )

        self.apply(weights_init_dcgan)

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h)
        h = h.view(-1, 256, 4, 4)
        h = self.up(h)
        out = self.to_map(h)  # [-1,1]
        return out

# Discriminator: mirror image GAN — Conv stride2 pyramid 64→128→128→256,
# LeakyReLU(0.2), Flatten, Dropout(0.4), Dense(1) with sigmoid.
class CGAN_Discriminator(nn.Module):
    def __init__(self, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape
        self.label_fc = nn.Linear(num_categories, H * W)

        def down(in_c, out_c, s=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=s, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            down(2, 64, s=1),  # “normal” conv
            down(64, 128, s=2),
            down(128, 128, s=2),
            down(128, 256, s=2),
            # add more downs to compress to a small map
            down(256, 256, s=2), # 16×64 -> 8×32
            down(256, 512, s=2), # 8×32 -> 4×16
            down(512, 512, s=2), # 4×16 -> 2×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512 * 2 * 8, 1),
            nn.Sigmoid()
        )

        self.apply(weights_init_dcgan)

    def forward(self, spec, y):
        B = spec.size(0)
        label_map = self.label_fc(y).view(B, 1, *self.spec_shape).to(spec.device)
        x = torch.cat([spec, label_map], dim=1)
        f = self.features(x)
        return self.classifier(f)

# ===============================================================================
# 3) GENERATION / SAVING
# ===============================================================================
def tanh_to_log1p_range(x_tanh, scale=5.0):
    # image GAN emits [-1,1]; map to a plausible log-mel magnitude range
    # log_mel ≈ [0, ~10] for many datasets; use simple affine to [0, 2*scale]
    return (x_tanh + 1) * scale

def generate_audio_gan(generator, category_idx, num_samples, device, sample_rate=22050):
    generator.eval()
    C = generator.num_categories
    z = torch.randn(num_samples, generator.latent_dim, device=device)
    y = F.one_hot(torch.tensor([category_idx], device=device), num_classes=C).float()
    y = y.repeat(num_samples, 1)
    with torch.no_grad():
        x = generator(z, y)                        # [-1,1] in shape [B,1,128,512]
        log_spec = tanh_to_log1p_range(x)          # map to nonnegative “log” range

    mel = torch.expm1(log_spec.squeeze(1))         # [B,128,512]
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=1024//2 + 1, n_mels=128, sample_rate=sample_rate
    ).to(device)
    linear_mag = inv_mel(mel)
    griff = torchaudio.transforms.GriffinLim(
        n_fft=1024, hop_length=256, win_length=1024, n_iter=32
    ).to(device)
    wav = griff(linear_mag)                        # [B, T]
    return wav.cpu()

def save_wav(wav, sr, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = wav[0].detach().cpu().numpy() if wav.dim()==2 else wav.detach().cpu().numpy()
    if w.ndim == 1: sf.write(path, w, sr)
    elif w.ndim == 2 and w.shape[0] == 1: sf.write(path, w.squeeze(0), sr)
    else: sf.write(path, w.T, sr)
    print("Saved:", path)

# ===============================================================================
# 4) TRAINING — image GAN hyperparameters
# ===============================================================================
def train_gan(generator, discriminator, dataloader, device, categories, epochs, latent_dim):
    optG = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    os.makedirs("gan_generated_audio", exist_ok=True)
    os.makedirs("gan_spectrogram_plots", exist_ok=True)

    for epoch in range(1, epochs+1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for real_specs, labels in loop:
            real_specs = real_specs.to(device)
            labels = labels.to(device)
            B = real_specs.size(0)
            real = torch.ones(B, 1, device=device)
            fake = torch.zeros(B, 1, device=device)

            # Train D
            optD.zero_grad()
            out_real = discriminator(real_specs, labels)
            loss_real = criterion(out_real, real)

            z = torch.randn(B, latent_dim, device=device)
            fake_specs = generator(z, labels)
            out_fake = discriminator(fake_specs.detach(), labels)
            loss_fake = criterion(out_fake, fake)
            lossD = 0.5*(loss_real + loss_fake)
            lossD.backward()
            optD.step()

            # Train G
            optG.zero_grad()
            out_g = discriminator(fake_specs, labels)
            lossG = criterion(out_g, real)  # generator tries to get D=1
            lossG.backward()
            optG.step()

            loop.set_postfix(loss_D=lossD.item(), loss_G=lossG.item())

        # Save plots/audio every 10 epochs (like the image script cadence)
        if epoch % 10 == 0:
            generator.eval()
            fig, axes = plt.subplots(1, len(categories), figsize=(4*len(categories), 3))
            if len(categories) == 1: axes = [axes]
            for i, name in enumerate(categories):
                y = F.one_hot(torch.tensor([i], device=device), num_classes=len(categories)).float()
                z = torch.randn(1, latent_dim, device=device)
                with torch.no_grad():
                    spec = generator(z, y)
                axes[i].imshow(spec.squeeze().detach().cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
                axes[i].set_title(f"{name} (ep {epoch})"); axes[i].axis('off')
            plt.tight_layout(); plt.savefig(f"gan_spectrogram_plots/epoch_{epoch:03d}.png"); plt.close(fig)

            for i, name in enumerate(categories):
                wav = generate_audio_gan(generator, i, 1, device)
                save_wav(wav, 22050, f"gan_generated_audio/{name}_ep{epoch}.wav")

            torch.save(generator.state_dict(), f"generator_model_{epoch:03d}.pth")
            generator.train()

# ===============================================================================
# 5) MAIN
# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./Train", help="Root folder with class subfolders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)  # match image GAN
    parser.add_argument("--latent_dim", type=int, default=100)  # match image GAN
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print("Using device:", device)

    categories = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    if not categories:
        raise RuntimeError(f"No class folders under {args.data}")
    print("Categories:", categories)

    ds = TrainAudioSpectrogramDataset(args.data, categories)
    nw = 0 if os.name == "nt" else 2
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, drop_last=True)

    G = CGAN_Generator(args.latent_dim, len(categories)).to(device)
    D = CGAN_Discriminator(len(categories)).to(device)
    print(G); print(D)

    train_gan(G, D, dl, device, categories, args.epochs, args.latent_dim)
