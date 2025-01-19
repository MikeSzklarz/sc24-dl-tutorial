export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export MASTER_ADDR_FALLBACK=$(hostname -I | awk '{print $1}')

python test_socket.py