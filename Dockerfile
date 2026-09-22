FROM vllm/vllm-openai:qwen38-flash-next
# FROM vllm/vllm-openai:v0.28.0-ubuntu2404

COPY tq-quant /tmp/tq-quant

RUN cd /tmp/tq-quant && \
    pip install --no-cache-dir --no-deps .

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# RUN pip install --no-cache-dir --upgrade \
#     git+https://github.com/huggingface/transformers.git

RUN ldd --version | head -1 && \
    python3 -c "import tq; print('TQ import OK:', tq.__file__)" && \
    python3 -c "import vllm; print('vLLM:', vllm.__version__, vllm.__file__)" 
    #  && \
    # python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('textclf/Qwen3.8-Flash-Next-TQ-4bit', trust_remote_code=True); print(type(c), c.model_type)"

ENV VLLM_WSL2_ENABLE_PIN_MEMORY=1

ENTRYPOINT []