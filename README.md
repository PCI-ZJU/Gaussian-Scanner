This is the main reconstruction code of the "Gaussian scanner: Low-cost and high-throughput 3D scanner based on differentiable rendering"

# Installation

```bash
# download
git clone https://github.com/PCI-ZJU/Gaussian-Scanner.git

conda env create --file environment.yml
conda activate surfel_splatting
```

# Datasets
The data will be available on [here](https://drive.google.com/drive/folders/1Xm55u31-GVZrwEsoHB1IB7p6A-MmzuF_?usp=drive_link)


# Overall pipeline

```bash
python train.py -s <path to dataset> -m output/<dataset_type> -r 2
