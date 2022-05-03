import pycuda.driver as cuda
import pycuda.autoinit

MAX_BYTES = 2 ** 30 #不设置为2 ** 31 - 1，而是确保为2的整数倍，从而提高对GPU的利用效率

#新增，待检查
def safe_cuda_malloc(num_bytes):
    if num_bytes > MAX_BYTES:
        print('Error! Cannot allocate that much GPU memory!')
        exit()
    return cuda.mem_alloc(int(num_bytes))

#新增，待检查
def safe_cuda_memcpy_htod(host_data):
    device_data = safe_cuda_malloc(host_data.nbytes)
    cuda.memcpy_htod(device_data, host_data)
    return device_data

# 检查过一遍，需检查第二遍
# 获得每个block共享内存最大能到多少bytes
def get_max_shmem_bytes_per_block():
    dev = pycuda.autoinit.device
    return dev.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
