import train
import socket
from contextlib import closing
import os

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


MASTER_ADDR = '127.0.0.1'
MASTER_PORT = find_free_port()
# set up the master's ip address so this child process can coordinate
os.environ['MASTER_ADDR'] = MASTER_ADDR
print(f"{MASTER_ADDR=}")
os.environ['MASTER_PORT'] = MASTER_PORT
print(f"{MASTER_PORT}")

os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_IB_DISABLE'] = '1'



# Configure the worker
train.config["n_workers"] = 2
train.config["rank"] = 0 # number of this worker in [0,4).

# Override some hyperparameters to train PowerSGD
train.config["optimizer_scale_lr_with_factor"] = 2  # workers
train.config["optimizer_reducer"] = "RankKReducer"
train.config["optimizer_reducer_rank"] = 2
train.config["optimizer_memory"] = True
train.config["optimizer_reducer_reuse_query"] = True
train.config["optimizer_reducer_n_power_iterations"] = 0

# You can customize the outputs of the training script by overriding these members
#train.output_dir = "choose_a_directory"
#train.log_info = your_function_pointer
#train.log_metric = your_metric_function_pointer

# Start training
train.main()