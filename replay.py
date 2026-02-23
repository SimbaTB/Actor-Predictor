import torch
import numpy as np

# 从numpy数据类型对象到pytorch数据类型的映射表
TYPE = {
    np.dtype('float32'): torch.float32,     # 这里浮点数统一用float32，因为模型参数就是float32
    np.dtype('float64'): torch.float32,
    np.dtype('float16'): torch.float32,
    np.dtype('int8'): torch.int8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('uint8'): torch.uint8,
    np.dtype('bool_'): torch.bool,
}

vars_to_save = ["capacity", "seq_len", "store_device", "target_device", "size", "ptr", "obs_shape",
                "act_shape", "obs", "action", "reward", "con", "is_valid_start"]

class ReplayBuffer:
    def __init__(self, capacity, obs_space, act_space, seq_len, store_device, target_device):
        '''
        环形经验回放缓冲区，每次采样一批连续的序列
        Args:
            capacity: 缓冲区容量
            obs_space: 观测空间(gym.spaces)
            act_space: 动作空间(gym.spaces)
            seq_len: 采样序列长度
            store_device: 数据存放设备
            target_device: 采样数据到指定设备
        '''
        self.capacity = capacity
        self.seq_len = seq_len
        self.store_device = store_device
        self.target_device = target_device
        self.size = 0
        self.ptr = 0

        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape
        self.obs = torch.zeros((capacity,) + self.obs_shape, dtype=TYPE[obs_space.dtype], device=store_device)
        self.action = torch.zeros((capacity,) + self.act_shape, dtype=TYPE[act_space.dtype], device=store_device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float, device=store_device)
        self.con = torch.zeros((capacity, 1), dtype=torch.float, device=store_device)
        self.is_valid_start = torch.zeros((capacity,), dtype=torch.bool, device=store_device)

    def add(self, obs, action, reward, terminated):
        assert obs.shape == self.obs_shape, obs.shape

        # ===== 添加数据 =====
        self.obs[self.ptr] = torch.tensor(obs).to(self.store_device)
        self.action[self.ptr] = torch.tensor(action).to(self.store_device)
        self.reward[self.ptr] = float(reward)
        self.con[self.ptr] = float(not terminated)
        
        # ===== 更新valid starts =====
        # 这里就不考虑环绕情况了，过于复杂而且概率很低（capacity远大于seq_len）
        # 因此，is_valid_start中最后的seq_len-1个元素恒为False
        p = self.ptr + 1 - self.seq_len

        if p >=0:
            # 求和小于seq_len - 1，说明中间有0，即出现间断
            # 这里减1,因为允许采样片段最后一个时间步con=0
            if self.con[p : p+self.seq_len-1].sum() < self.seq_len - 1:
                self.is_valid_start[p] = False
            else:
                self.is_valid_start[p] = True

        #由于这里使用环形缓冲区，当前episode数据只到self.ptr，下标在[p+1, self.ptr]范围内的都不能作为采样起始点
        self.is_valid_start[max(0,p+1) : self.ptr+1] = False

        # ===== 更新指针和大小 =====
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device=None):
        target_device = device or self.target_device
        # ===== 从is_valid_start中抽取下标 =====
        valid_starts = torch.where(self.is_valid_start)[0]
        N = valid_starts.shape[0]
        #print("valid_starts:", valid_starts)

        if N < batch_size:
            return None
        
        rand_idx = torch.randint(0, N, (batch_size,), device=self.store_device)
        starts = valid_starts[rand_idx]

        # ===== 构造索引矩阵 =====
        # is_valid_start中只有下标小于等于capacity - seq_length的才会被标记为True，此处无需担心溢出
        arange = torch.arange(self.seq_len, device=self.store_device)  # (seq_len,)
        indices = starts.unsqueeze(1) + arange  #构造索引矩阵 (batch_size, 1) + (seq_len,) → (batch_size, seq_len)

        # ===== 读取数据 =====
        batch = {
            "obs": self.obs[indices].to(target_device),
            "action": self.action[indices].to(target_device),
            "reward": self.reward[indices].to(target_device),
            "con": self.con[indices].to(target_device)
        }
        return batch
    
    def get_size(self):
        return self.size

    def state_dict(self):
        return {key: getattr(self, key) for key in vars_to_save}
    
    def load_state_dict(self, state_dict):
        for key in vars_to_save:
            setattr(self, key, state_dict[key])