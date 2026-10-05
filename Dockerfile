FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425 AS llama-build

ARG LLAMA_CPP_COMMIT=95c409c13625a23da2aa37270339ce9179215a18

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git /src/llama.cpp \
    && git -C /src/llama.cpp checkout --detach "${LLAMA_CPP_COMMIT}"
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_NCCL=OFF \
        -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    && cmake --build /src/llama.cpp/build --config Release --target llama-server --parallel

FROM nvcr.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04@sha256:8bbc6e304b193e84327fa30d93eea70ec0213b808239a46602a919a479a73b12 AS cuda-runtime

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/tmp/chimera-home \
    XDG_CACHE_HOME=/tmp/chimera-cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 algorithm \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin algorithm

WORKDIR /opt/app

COPY requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /opt/app/requirements.txt

COPY --from=cuda-runtime /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudart.so* /usr/local/cuda/lib64/
COPY --from=cuda-runtime /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcublas.so* /usr/local/cuda/lib64/
COPY --from=cuda-runtime /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcublasLt.so* /usr/local/cuda/lib64/
COPY --from=llama-build /src/llama.cpp/build/bin/llama-server /opt/app/bin/llama-server

COPY --chown=10001:10001 inference.py /opt/app/inference.py
COPY --chown=10001:10001 src/ /opt/app/src/
COPY --chown=10001:10001 configs/ /opt/app/configs/
COPY --chown=10001:10001 templates/ /opt/app/templates/
COPY --chown=10001:10001 resources/task3_head.json /opt/app/resources/task3_head.json
# Task 1 and Task 2 staged policies. resources/ is copied file by file here,
# so a new resource must be named explicitly or policy.py fails at first use.
# The agentic policy's thresholds for Tasks 1 and 2. The frozen task*_policy
# files below are no longer on the inference path but are kept for reference.
# ⚠️ The decision graphs. Tasks 1 and 2 are decided by these; without them the
# container raises on the first case.
COPY --chown=10001:10001 resources/task1_graph.yaml /opt/app/resources/task1_graph.yaml
COPY --chown=10001:10001 resources/task2_graph.yaml /opt/app/resources/task2_graph.yaml
COPY --chown=10001:10001 resources/agentic_task1.yaml /opt/app/resources/agentic_task1.yaml
COPY --chown=10001:10001 resources/task1_policy.yaml /opt/app/resources/task1_policy.yaml
COPY --chown=10001:10001 resources/task2_policy.yaml /opt/app/resources/task2_policy.yaml
USER 10001:10001

ENTRYPOINT ["python", "/opt/app/inference.py"]
