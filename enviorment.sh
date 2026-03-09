echo "Install pytorch (CUDA 11.7, 官方源)"
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    -f https://download.pytorch.org/whl/cu117

pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
