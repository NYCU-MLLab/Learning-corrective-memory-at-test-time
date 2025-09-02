## 1. Install the CUDA kernel for Longhorn and Mamba
```
git clone https://github.com/Cranial-XIX/longhorn_cuda.git
cd longhorn_cuda
chmod +x rebuild.sh && bash rebuild.sh
```

## 2. Data Download
Please download the file throught google drive link below and unzip to ./datasets:
- [Google Link](https://drive.google.com/drive/folders/19KwnEuuf3sfST8nIfoRlUZX6r8GnEGnx?usp=drive_link)

## 3. Train on Slimpajama

Prepare the dataset
```
python Slim/prepare.py
```

Then run with:
```
bash run.sh MODEL
```
where `MODEL` can be any of `longhorn, ttt, preco`.

## 3. Testing
```
python infer.py
```

## 4. Evaluation







