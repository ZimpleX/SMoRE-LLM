set -x
MY_ENV=smore_train
PYTHON_VERSION=3.11.11
CUDA_VERSION=12.8
TORCH_VERSION=2.7.0
TORCH_CUDA_VERSION=cu128
source ~/miniconda3/etc/profile.d/conda.sh
source activate base
# Ensure Anaconda channel Terms of Service are accepted non-interactively
# (prevents `conda create` from failing in automated/scripted runs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda activate $MY_ENV || conda create --name $MY_ENV python=$PYTHON_VERSION --yes
conda activate $MY_ENV
# echo $CONDA_DEFAULT_ENV
# if [ "$CONDA_DEFAULT_ENV" != "$MY_ENV" ];
# then
#     echo "ERROR: Failed to create environment $MY_ENV"
#     exit
# fi
# echo $(which nvcc)
# if [ "$(which nvcc | grep /cuda-${CUDA_VERSION})" == "" ];
# then
#     echo "ERROR: NVCC for CUDA $CUDA_VERSION not found. You may need log into a GPU node first."
#     exit
# fi
conda install --yes pip
PIP_PATH=$(which pip)
echo $PIP_PATH
# if [ "$PIP_PATH" != "/home/${USER}/miniconda3/envs/${MY_ENV}/bin/pip" ];
# then
#     echo "ERROR: Could not install pip into conda"
#     exit
# fi
pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cu128
if [ "$(pip show torch | grep Version)" != "Version: ${TORCH_VERSION}+${TORCH_CUDA_VERSION}" ];
then
    echo "ERROR: Could not install torch==${TORCH_VERSION}+${TORCH_CUDA_VERSION}"
    exit
fi
pip install deepspeed==0.15.4
pip install ninja==1.11.1.4
pip install huggingface-hub==0.27.0
pip install datasets==3.0.1
pip install accelerate==1.0.1
pip install transformers==4.47.1
pip install evaluate==0.4.3
pip install gpustat
pip install matplotlib==3.10.0 seaborn==0.13.2
pip install pandas==2.2.3
pip install scipy==1.14.1 scikit-learn==1.6.0
pip install wandb==0.19.1
pip install starlette==0.41.3 sse-starlette==2.1.3 fastapi==0.115.6
pip install fire==0.7.0
pip install gradio==4.44.1 gradio-client==1.3.0
pip install protobuf==5.26.1
pip install tiktoken==0.8.0
pip install typer==0.15.1 typing-extensions==4.15.0
pip install trl==0.9.6
pip install tyro==0.8.14
pip install uvicorn==0.34.0
cd ..
cd peft && pip install --upgrade -e .
cd ..
cd model_moe && pip install --upgrade -e .
cd ..
pip install debugpy==1.8.11
cd LLaMA-Factory && pip install -e ".[torch,metrics]" --no-deps --no-build-isolation
# cd ..
# pip install flash-attn==2.7.2.post1 --no-build-isolation