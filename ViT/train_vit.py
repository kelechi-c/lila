import torch
import wandb
import os
import gc
from torch import optim
from torch import nn
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from safetensors.torch import save_model
from .utils import config, count_params
from .vit import vit
from .image_data import train_loader


criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.AdamW(params=vit.parameters(), lr=config.lr)
scaler = GradScaler()


param_count = count_params(vit)
print(param_count)

epochs = config.epoch_count


# initilaize wandb
wandb.login()
train_run = wandb.init(project="vit_mini", name="vit_1")
wandb.watch(vit, log_freq=100)


if os.path.exists(config.model_outpath) is not True:
    os.mkdir(config.model_outpath)

output_path = os.path.join(os.getcwd(), config.model_outpath)

torch.cuda.empty_cache()
gc.collect()


def training_loop(model=vit, train_loader=train_loader, epochs=epochs, config=config):
    model.train()
    train_loss = 0.0

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        torch.cuda.empty_cache()
        print(f"Training epoch {epoch+1}")

        for x, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(config.device)
            label = label.to(config.device)

            # every iterations
            torch.cuda.empty_cache()
            gc.collect()

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = model(image)
                train_loss = criterion(output, label.long())
                train_loss = train_loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place

                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

            wandb.log({"loss": train_loss})

        print(f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}")

        print(f"Epoch @ {epoch} complete!")

    print(f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}")

    safe_tensorfile = save_model(model, config.safetensor_file)

    torch.save(model.state_dict(), os.path.join(output_path, f"{config.model_file}"))


training_loop()
torch.cuda.empty_cache()
gc.collect()


print("vit-mini pretraining complete")
# Ciao
