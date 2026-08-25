FROM python:3.11-slim-bookworm

RUN python -m venv /opt/inspect_tool_support \
    && /opt/inspect_tool_support/bin/pip install --no-cache-dir \
       inspect-tool-support \
       jupyter_client==8.6.3 \
       jupyter_kernel_gateway==3.0.1 \
       numpy==1.26.4 \
       pandas==2.3.3 \
       scipy==1.16.3 \
       sympy==1.14.0 \
       networkx==3.5 \
       matplotlib==3.10.7 \
    && /opt/inspect_tool_support/bin/inspect-tool-support post-install

ENV PATH="/opt/inspect_tool_support/bin:${PATH}" \
    HOME=/tmp \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
USER 65532:65532
CMD ["tail", "-f", "/dev/null"]
