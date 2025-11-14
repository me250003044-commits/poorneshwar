# GAN Audio Generation - Improved for Local Run

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# 0. UTIL: weight init
# -----------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    elif classname.find('BatchNorm') != -1:
        try:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        except Exception:
            pass

# ===============================================================================
# 1. DATASET DEFINITION
# ===============================================================================
class TrainAudioSpectrogramDataset(Dataset):
    """
    Loads wav files from class subfolders, computes log-mel spectrograms and
    returns (spec, one-hot-label).
    Directory layout expected: root_dir/<class_name>/*.wav
    """
    def __init__(self, root_dir, categories=None, max_frames=512, fraction=1.0, sample_rate=22050):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.sample_rate = sample_rate

        if categories is None:
            # auto-detect categories as subdirectories
            categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.categories = categories
        self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}

        self.file_list = []
        for cat_name in self.categories:
            cat_dir = os.path.join(root_dir, cat_name)
            if not os.path.exists(cat_dir):
                continue
            files_in_cat = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith('.wav')]
            num_to_sample = int(len(files_in_cat) * fraction)
            if num_to_sample == 0:
                continue
            sampled_files = random.sample(files_in_cat, num_to_sample)
            label_idx = self.class_to_idx[cat_name]
            self.file_list.extend([(file_path, label_idx) for file_path in sampled_files])

        if len(self.file_list) == 0:
            raise RuntimeError(f"No wav files found in {root_dir} for categories: {self.categories}")

        # Pre-create mel transform (doesn't hold device-specific tensor state)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=256, n_mels=128
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]

        # ---- LOAD WAV SAFELY (no torchaudio.load) ----
        wav, sr = sf.read(path)  # Use soundfile instead of torchaudio
        wav = torch.tensor(wav).float()

        # Convert to mono
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        wav = wav.unsqueeze(0)  # (1, N)

        # Resample
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # ---- MEL SPECTROGRAM ----
        mel = self.mel_transform(wav)
        logmel = torch.log1p(mel)

        _, _, T = logmel.shape
        if T < self.max_frames:
            logmel = F.pad(logmel, (0, self.max_frames - T))
        else:
            logmel = logmel[:, :, :self.max_frames]

        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()

        return logmel, label_vec

# ===============================================================================
# 2. MODEL DEFINITIONS
# ===============================================================================
class CGAN_Generator(nn.Module):
    """Generator: maps (z, y) -> log-mel spectrogram"""
    def __init__(self, latent_dim, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape

        hidden = 256
        self.fc = nn.Linear(latent_dim + num_categories, hidden * 8 * 32)
        self.unflatten_shape = (hidden, 8, 32)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # final activation: use softplus to ensure positive outputs (log1p domain)
            nn.Softplus()
        )

        # init
        self.apply(weights_init)

    def forward(self, z, y):
        # z: (B, latent_dim), y: (B, num_categories)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        h = torch.cat([z, y], dim=1)
        h = self.fc(h)
        h = h.view(-1, *self.unflatten_shape)
        fake_spec = self.net(h)
        return fake_spec


class CGAN_Discriminator(nn.Module):
    """Discriminator: takes spectrogram + label-map -> logit"""
    def __init__(self, num_categories, spec_shape=(128, 512)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape

        # better label embedding: small MLP then reshape
        self.label_embedding = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(True),
            nn.Linear(256, H * W)
        )

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            # input channels = 2 (spec + label map)
            nn.utils.spectral_norm(nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),

            # final conv to single logit
            nn.utils.spectral_norm(nn.Conv2d(256, 1, kernel_size=(8, 32), stride=1, padding=0))
        )

        self.apply(weights_init)

    def forward(self, spec, y):
        # spec: (B,1,H,W), y: (B, num_categories)
        label_map = self.label_embedding(y).view(-1, 1, *self.spec_shape).to(spec.device)
        h = torch.cat([spec, label_map], dim=1)
        logit = self.net(h)
        return logit.view(-1, 1)

# ===============================================================================
# 3. GENERATE / SAVE UTILITIES
# ===============================================================================

def generate_audio_gan(generator, category_idx, num_samples, device, sample_rate=22050):
    generator.eval()
    num_categories = generator.num_categories
    latent_dim = generator.latent_dim

    # Prepare label and noise
    y = F.one_hot(torch.tensor([category_idx]), num_classes=num_categories).float().to(device)
    # repeat to match number of samples
    y = y.repeat(num_samples, 1)
    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        log_spec_gen = generator(z, y)

    # convert back from log1p domain
    spec_gen = torch.expm1(log_spec_gen)
    spec_gen = spec_gen.squeeze(1)  # (B, n_mels, frames)

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1, n_mels=128, sample_rate=sample_rate
    ).to(device)
    linear_spec = inverse_mel(spec_gen)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=1024, hop_length=256, win_length=1024, n_iter=64
    ).to(device)

    waveform = griffin(linear_spec)
    return waveform.cpu()


def save_wav(wav, sample_rate, filename):
    # FIXED: Use soundfile instead of torchaudio to avoid FFmpeg dependency
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Handle different tensor shapes and ensure proper format
    if wav.dim() == 3:  # (batch, channels, samples)
        wav_to_save = wav[0]
    elif wav.dim() == 2:  # (channels, samples)
        wav_to_save = wav
    else:  # (samples,)
        wav_to_save = wav.unsqueeze(0)
    
    # Convert to numpy and ensure proper shape for soundfile
    wav_np = wav_to_save.numpy()
    
    # Make sure the audio data is in the right format
    # Normalize to prevent clipping and ensure valid range
    if wav_np.size > 0:
        max_val = np.max(np.abs(wav_np))
        if max_val > 0:
            wav_np = wav_np / max_val * 0.9  # Scale to 90% to avoid clipping
    
    # Ensure the data is float32 and properly shaped
    wav_np = wav_np.astype(np.float32)
    
    # If stereo (2 channels), transpose to (samples, channels) for soundfile
    if wav_np.shape[0] == 2:
        wav_np = wav_np.T
    
    # For mono audio, ensure it's 1D array
    if wav_np.shape[0] == 1:
        wav_np = wav_np.flatten()
    
    try:
        # Save using soundfile - much more reliable than torchaudio
        sf.write(filename, wav_np, sample_rate)
        print(f"Saved to {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        # Fallback: try to save with a different name
        try:
            backup_name = filename.replace('.wav', '_backup.wav')
            sf.write(backup_name, wav_np, sample_rate)
            print(f"Saved backup to {backup_name}")
        except:
            print("Could not save audio file")

# ===============================================================================
# 4. TRAINING LOOP
# ===============================================================================

def train_gan(generator, discriminator, dataloader, device, categories, epochs, lr, latent_dim, play_audio=False):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs("gan_generated_audio", exist_ok=True)
    os.makedirs("gan_spectrogram_plots", exist_ok=True)

    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for real_specs, labels in loop:
            real_specs = real_specs.to(device)
            labels = labels.to(device)
            batch_size = real_specs.size(0)

            real_labels_tensor = torch.ones(batch_size, 1, device=device)
            fake_labels_tensor = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(real_specs, labels)
            loss_D_real = criterion(real_output, real_labels_tensor)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_specs = generator(z, labels)

            fake_output = discriminator(fake_specs.detach(), labels)
            loss_D_fake = criterion(fake_output, fake_labels_tensor)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            output = discriminator(fake_specs, labels)
            loss_G = criterion(output, real_labels_tensor)
            loss_G.backward()
            optimizer_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        # end epoch: generate samples and save
        print(f"\n--- Generating Samples for Epoch {epoch} ---")
        generator.eval()

        fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
        if len(categories) == 1:
            axes = [axes]

        for cat_idx, cat_name in enumerate(categories):
            y_cond = F.one_hot(torch.tensor([cat_idx]), num_classes=generator.num_categories).float().to(device)
            z_sample = torch.randn(1, generator.latent_dim).to(device)
            with torch.no_grad():
                spec_gen_log = generator(z_sample, y_cond)

            spec_gen_log_np = spec_gen_log.squeeze().cpu().numpy()
            axes[cat_idx].imshow(spec_gen_log_np, aspect='auto', origin='lower')
            axes[cat_idx].set_title(f'{cat_name} (Epoch {epoch})')
            axes[cat_idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'gan_spectrogram_plots/epoch_{epoch:03d}.png')
        plt.close(fig)

        for cat_idx, cat_name in enumerate(categories):
            wav = generate_audio_gan(generator, cat_idx, 1, device)
            fname = f"gan_generated_audio/{cat_name}_ep{epoch}.wav"
            save_wav(wav, sample_rate=22050, filename=fname)
            if play_audio:
                try:
                    from IPython.display import Audio, display
                    display(Audio(data=wav.numpy()[0], rate=22050))
                except Exception:
                    pass

        generator.train()
        print("--- End of Sample Generation ---\n")

# ===============================================================================
# 5. MAIN
# ===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='train/', help='Root folder containing class subfolders with wav files')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--play', action='store_true', help='Attempt to play generated audio if running in a notebook')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    # discover categories
    TRAIN_PATH = args.data  # default is 'train/'
    train_categories = sorted([d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))])
    NUM_CATEGORIES = len(train_categories)
    if NUM_CATEGORIES == 0:
        raise RuntimeError(f"No class folders found in {TRAIN_PATH}. Expected layout: {TRAIN_PATH}/<label>/*.wav")

    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CATEGORIES} categories: {train_categories}")

    train_dataset = TrainAudioSpectrogramDataset(root_dir=TRAIN_PATH, categories=train_categories)
    # On Windows, num_workers should often be 0
    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    generator = CGAN_Generator(args.latent_dim, NUM_CATEGORIES).to(DEVICE)
    discriminator = CGAN_Discriminator(NUM_CATEGORIES).to(DEVICE)

    train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=train_loader,
        device=DEVICE,
        categories=train_categories,
        epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        play_audio=args.play
    )
