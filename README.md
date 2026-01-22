# $${\color{white}ComfyUI-}{\color{orange}R}{\color{yellow}T}{\color{white}-HeartMuLa}$$

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
```

___

### $${\color{yellow}Model \ Sources}$$

Explore the technical foundations and official repositories for the models used in this project:

* **GitHub Repository:** [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib)
* **Technical Paper:** [ArXiv: 2601.10547](https://arxiv.org/abs/2601.10547)
* **Official Demo:** [HeartMuLa Project Page](https://heartmula.github.io/)

#### **Hugging Face Model Hub:**
* **Main Model (3B):** [HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B)
* **Codec:** [HeartCodec-oss](https://huggingface.co/HeartMuLa/HeartCodec-oss)
* **Transcriptor:** [HeartTranscriptor-oss](https://huggingface.co/HeartMuLa/HeartTranscriptor-oss)

---

### $${\color{yellow}Credits}$$

Special thanks to the **HeartMuLa** team for providing the open-source weights and research that make these nodes possible.

* **HeartMuLa Organization:** [Hugging Face Profile](https://huggingface.co/HeartMuLa)

