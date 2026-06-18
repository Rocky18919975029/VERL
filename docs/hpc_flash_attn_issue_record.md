# ACD HPC flash-attn Issue Record

This document records the observed flash-attn failure sequence on the ACD HPC cluster while attempting to run verl GRPO/RLVR training with `flash_attention_2`.

## Training Job Context

The target training job was a verl GRPO/RLVR run using the local Qwen2.5-7B base model:

```bash
sbatch --export=ALL,METHOD=baseline,MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-7b-local \
  examples/grpo_trainer/submit_qwen2_5_7b_entropy_h100.slurm
```

The job was intended to run from:

```text
/data/user/zhongal/VERL
```

The local model path used for the job was:

```text
/data/user/zhongal/.cache/qwen2.5-7b-local
```

The data paths used by the entropy/GRPO script were:

```text
Training file: /data/user/zhongal/data/entropy/dapo-math-17k.parquet
Validation file: /data/user/zhongal/data/entropy/aime-2024.parquet
```

## HPC Environment

The relevant HPC environment reported during the debugging session was:

```text
Cluster login host: ACD-Manage-3
Slurm partition: acd_u
GPU nodes: 8 GPUs per node
CPU limit observed: no more than 12 CPUs per GPU
Compute-node network: unavailable
```

The conda environment used by the job was:

```text
/data/user/zhongal/.conda/envs/verl
```

The Python and PyTorch environment reported:

```text
Python: 3.12.13
Torch: 2.8.0+cu128
torch.version.cuda: 12.8
torch._C._GLIBCXX_USE_CXX11_ABI: True
Python tag: cp312
```

The relevant modules available on the HPC included:

```text
cuda/12.8
gcc/13.3
glibc/2.32
glibc/2.34
glibc/2.35
```

## Initial Training Failure

The training job reached the verl/Ray worker path and failed while importing flash-attn:

```text
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
(required by /data/user/zhongal/.conda/envs/verl/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so)
```

This indicated that the installed `flash_attn_2_cuda` extension required `GLIBC_2.32`, while the runtime was loading `/lib64/libc.so.6`.

## Preflight Import Failure

A Slurm preflight check was added to import `flash_attn` before launching training. The job then failed before training started:

```text
failed to import flash_attn: /lib64/libc.so.6: version `GLIBC_2.32' not found
(required by /data/user/zhongal/.conda/envs/verl/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so)
```

The same error occurred outside the Ray worker context, confirming that the failure was not specific to Ray worker environment propagation.

## Prebuilt flash-attn Wheel Attempt

The environment ABI and Python tag were checked:

```bash
python - <<'PY'
import sys, torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("python tag:", f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
```

The output was:

```text
torch: 2.8.0+cu128
torch cuda: 12.8
abi: True
python tag: cp312
```

The following prebuilt wheel was downloaded from the flash-attention GitHub release assets:

```text
flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

The download and installation completed:

```text
Successfully installed flash-attn-2.8.3.post1
```

Testing the installed wheel produced:

```text
flash_attn_2_cuda: /data/user/zhongal/.conda/envs/verl/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so
/data/user/zhongal/.conda/envs/verl/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: /lib64/libc.so.6: version `GLIBC_2.32' not found
```

The Python import then failed:

```text
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
(required by /data/user/zhongal/.conda/envs/verl/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so)
```

## glibc Module Inspection

The `glibc/2.32` module was inspected:

```bash
module show glibc/2.32
```

It reported:

```text
/data/modulefiles/tools/glibc/2.32:

module-whatis   {Loads glibc 2.32}
prepend-path    PATH /data/apps/glibc/2.32/bin:/data/apps/glibc/2.32/sbin
prepend-path    CPATH /data/apps/glibc/2.32/include
prepend-path    MANPATH /data/apps/glibc/2.32/share/man
setenv          GLIBC_ROOT /data/apps/glibc/2.32
```

After loading the module:

```bash
module purge
module load glibc/2.32
```

`ldd` resolved to the glibc module path:

```text
which ldd
/data/apps/glibc/2.32/bin/ldd

ldd --version | head -n 1
ldd (GNU libc) 2.32
```

The glibc 2.32 libc file existed at:

```text
/data/apps/glibc/2.32/lib/libc.so.6
```

## Runtime glibc Path Experiment

After loading `glibc/2.32` and `cuda/12.8`, the observed `LD_LIBRARY_PATH` included the glibc module library paths:

```text
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/data/apps/glibc/2.32/lib:/data/apps/glibc/2.32/lib:/data/apps/glibc/2.32/lib64
```

Running:

```bash
python - <<'PY'
import flash_attn
print("ok")
PY
```

resulted in:

```text
Segmentation fault (core dumped)
```

After this environment state, other commands in the same shell also failed. For example:

```bash
git pull origin main
```

resulted in:

```text
Segmentation fault (core dumped)
```

The shell was later reset by logging in again.

## Source Build Attempt on Login Node

An attempt was made to build `flash-attn==2.8.1` or `flash-attn==2.8.3.post1` from source in the conda environment.

The first source-build failure was due to an old default GCC:

```text
#error "You're trying to build PyTorch with a too old version of GCC. We need GCC 9 or later."
```

After loading a newer compiler:

```bash
module purge
module load gcc/13.3
module load cuda/12.8
source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/zhongal/.conda/envs/verl
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
```

the compiler version was:

```text
gcc (GCC) 13.3.0
```

The build then failed later during CUDA compilation with:

```text
Killed "$CICC_PATH/cicc"
ninja: build stopped: subcommand failed.
```

This occurred with parallel build settings such as:

```text
MAX_JOBS=16
```

and also later with:

```text
MAX_JOBS=1
```

The `MAX_JOBS=1` failure included:

```text
Command '['ninja', '-v', '-j', '1']' returned non-zero exit status 255.
```

The CUDA compilation command in the failure logs included:

```text
-arch compute_120
```

even though `TORCH_CUDA_ARCH_LIST="9.0"` had been set in one attempt.

## Source Archive Download Attempt

Because compute nodes do not have network access, the source archive was downloaded on the login node for use by a Slurm build job.

The first command:

```bash
pip download --no-binary flash-attn --no-deps -d /data/user/zhongal/packages flash-attn==2.8.3.post1
```

failed during build requirement discovery:

```text
ModuleNotFoundError: No module named 'torch'
```

This occurred inside pip's build-isolation environment.

The later command included `--no-build-isolation`:

```bash
pip download --no-build-isolation --no-binary flash-attn --no-deps \
  -d /data/user/zhongal/packages flash-attn==2.8.3.post1
```

## Slurm Build Script Attempt

A Slurm build script was added at:

```text
examples/grpo_trainer/build_flash_attn_h100.slurm
```

It was intended to compile flash-attn on an H100 compute node using the downloaded source archive.

The first submission failed due to the cluster CPU-per-GPU policy:

```text
sbatch: error: CPU 超限: 每 GPU 对应 CPU 不能超过 12 核
sbatch: error: 申请核数：32 ；申请卡数: 1 (每卡 32 核)
sbatch: error: Batch job submission failed: Unspecified error
```

The script was then changed from:

```text
#SBATCH --cpus-per-task=32
```

to:

```text
#SBATCH --cpus-per-task=12
```

The latest repository commit containing this change was:

```text
76988e81 Respect ACD CPU per GPU limit
```

## Relevant Repository Changes Made During Debugging

The following local/repository changes were made during the debugging process:

```text
e5d8e65e Add Qwen2.5 entropy GRPO smoke scripts
ae527b49 Unset ROCm visibility vars in H100 Slurm jobs
8bd47e62 Default H100 GRPO scripts to SDPA attention
2633819a Load GLIBC module for flash attention jobs
67e70c01 Propagate GLIBC paths to Ray workers
77ad4eab Build flash attention on H100 nodes
76988e81 Respect ACD CPU per GPU limit
```

The current state of the debugging was that the prebuilt flash-attn wheel installed successfully but failed at import time because of `GLIBC_2.32`, while source compilation on the login node failed during CUDA compilation with `cicc` being killed.
