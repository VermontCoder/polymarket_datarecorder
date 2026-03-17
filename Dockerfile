FROM vastai/pytorch:2.5.1-cuda12.1.1

RUN apt-get update && apt-get install -y \
    openssh-server \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

COPY test_pytorch_gpu.py /workspace/test_pytorch_gpu.py

RUN mkdir -p /run/sshd

COPY setup-ssh-key.sh /opt/setup-ssh-key.sh

RUN chmod +x /opt/setup-ssh-key.sh

COPY supervisord-extras.conf /etc/supervisor/conf.d/ssh-extras.conf

EXPOSE 22
EXPOSE 6006
EXPOSE 18080

ENV PORTAL_CONFIG=localhost:6006:6006:/tensorboard:TensorBoard
