# ComfyUI-RT-HeartMuLa

A custom node suite for ComfyUI.

## Installation

Follow these steps to get **ComfyUI-RT-HeartMuLa** up and running.

### $${\color{yellow}Step \ 1: \ Clone \ the \ Repository}$$
Open your terminal or command prompt, navigate to your `ComfyUI\custom_nodes` folder, and run:

```bash
git clone https://github.com/monnky/ComfyUI-RT-HeartMuLa

```
### $${\color{yellow}Step \ 2: \ Navigate \ to \ the \ Directory}$$
Change into the project folder:

```bash
cd ComfyUI-RT-HeartMuLa
```


### $${\color{yellow}Step \ 3: \ Install \ Dependencies}$$

First Try install only 3 dep's
```bash
pip install soundfile
pip install torchtune
pip install torchao
```

(if doesnt work, install all requirments)
Install the required Python packages:

```bash
pip install -r requirements.txt
```


### $${\color{yellow}Step \ 4: \ Download \ Model \ Files}$$

Navigate to your `ComfyUI/models` directory and use the Hugging Face CLI to download the required weights.

> [!TIP]
> Ensure you have the Hugging Face CLI installed (`pip install huggingface_hub`).

```bash
# Navigate to models folder
cd ComfyUI/models

# Download model weights to specific local directories
huggingface-cli download HeartMuLa/HeartMuLaGen --local-dir ./HeartMuLa
huggingface-cli download HeartMuLa/HeartMuLa-oss-3B --local-dir ./HeartMuLa/HeartMuLa-oss-3B
huggingface-cli download HeartMuLa/HeartCodec-oss --local-dir ./HeartMuLa/HeartCodec-oss
huggingface-cli download HeartMuLa/HeartTranscriptor-oss --local-dir ./HeartMuLa/HeartTranscriptor-oss
