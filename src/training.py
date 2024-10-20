import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import random_noise_generator, discriminator_loss, generator_loss
import os

def train(D, G, disc_opt, gen_opt, train_loader, batch_size, epochs=25, gen_input_size=100, device='cuda'):
    disc_losses = []
    gen_losses = []
    
    sample_size = 8
    fixed_samples = random_noise_generator(sample_size, gen_input_size)
    
    # Para guardar imágenes en ciertos pasos
    saved_images = []
    
    # Directorio donde se guardarán las imágenes generadas
    os.makedirs("generated_images", exist_ok=True)
    
    best_loss = float('inf')
    best_samples = None
    
    for epoch in range(epochs + 1):
        D.train()
        G.train()
        disc_loss_total = 0
        gen_loss_total = 0

        for train_x, _ in tqdm(train_loader):
            # Entrenamiento del discriminador
            disc_opt.zero_grad()
            train_x = train_x.to(device)
            real_out = D(train_x.float())

            disc_gen_in = random_noise_generator(batch_size, gen_input_size)
            disc_gen_out = G(disc_gen_in.float()).detach()
            fake_out = D(disc_gen_out.float())

            disc_loss = discriminator_loss(real_out, fake_out)
            disc_loss_total += disc_loss.item()
            disc_loss.backward()
            disc_opt.step()

            # Entrenamiento del generador
            gen_opt.zero_grad()
            gen_gen_in = random_noise_generator(batch_size, gen_input_size)
            gen_out = G(gen_gen_in.float())
            gen_disc_out = D(gen_out.float())

            gen_loss = generator_loss(gen_disc_out)
            gen_loss_total += gen_loss.item()
            gen_loss.backward()
            gen_opt.step()

        disc_losses.append(disc_loss_total)
        gen_losses.append(gen_loss_total)

        # Actualizar mejor pérdida y muestra de imágenes
        total_loss = disc_loss_total + gen_loss_total
        if total_loss < best_loss:
            best_loss = total_loss
            best_samples = G(fixed_samples.float()).detach().cpu()

        # Guardar imágenes solo en epochs seleccionados
        if epoch == 0 or epoch % (epochs // 4) == 0 or epoch == epochs:
            G.eval()
            samples = G(fixed_samples.float()).detach().cpu()
            G.train()

            fig, axes = plt.subplots(1, 8, figsize=(12, 12))
            plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Espaciado entre imágenes

            for i in range(8):
                axes[i].imshow(samples[i].permute(1, 2, 0) * 0.5 + 0.5)  # Desnormalizar imágenes para visualización
                axes[i].axis('off')
            
            # Guardar la figura en disco
            image_path = f"generated_images/epoch_{epoch}.png"
            plt.savefig(image_path)
            plt.close()
            
            saved_images.append(image_path)

        # Visualizar las imágenes cada 8 epochs
        if epoch % 8 == 0:
            G.eval()
            samples = G(fixed_samples.float()).detach().cpu()
            G.train()

            fig, axes = plt.subplots(1, 8, figsize=(12, 12))
            plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Espaciado entre imágenes

            for i in range(8):
                axes[i].imshow(samples[i].permute(1, 2, 0) * 0.5 + 0.5)  # Desnormalizar imágenes para visualización
                axes[i].axis('off')
            plt.show()

    # Guardar la mejor imagen generada al final
    if best_samples is not None:
        fig, axes = plt.subplots(1, 8, figsize=(12, 12))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        for i in range(8):
            axes[i].imshow(best_samples[i].permute(1, 2, 0) * 0.5 + 0.5)  # Desnormalizar
            axes[i].axis('off')
        
        best_image_path = "generated_images/best_epoch.png"
        plt.savefig(best_image_path)
        plt.close()

        saved_images.append(best_image_path)

    # Limitar el número de imágenes guardadas a 5 (incluyendo la mejor)
    saved_images = saved_images[:4] + [best_image_path]

    print(f"Imágenes guardadas en: {saved_images}")

