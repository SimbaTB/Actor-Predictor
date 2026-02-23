import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
import nets
import tools

class Predictor(nn.Module):
    def __init__(self, obs_dim, act_space, hidden_dim, layers_num):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_discrete = tools.is_discrete(act_space)
        self.act_dim = act_space.n if self.act_discrete else act_space.shape[0]
        self.act_shape = act_space.shape

        self.fc1 = nn.Linear(obs_dim + self.act_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, layers_num, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, obs, action, hx=None):
        B, T = obs.shape[:2]
        assert obs.shape == (B, T, self.obs_dim), obs.shape
        assert action.shape == (B, T, *self.act_shape), action.shape

        if self.act_discrete:
            action = F.one_hot(action, num_classes=self.act_dim).float()
        
        x = torch.cat([obs, action], dim=-1)
        x = F.silu(self.norm1(self.fc1(x)))
        x, hx = self.lstm(x, hx)
        pred = F.silu(self.norm2(self.fc2(x)))

        return pred, hx

class ActorContinuous(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_space, layers_num=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        scale = (act_space.high - act_space.low) / 2
        offset = (act_space.high + act_space.low) / 2
        # 注册后，才能在.to(device)时随对象一起转移到设备
        self.register_buffer('action_scale', torch.tensor(scale, dtype=torch.float))
        self.register_buffer('action_offset', torch.tensor(offset, dtype=torch.float))
        self.log2 = torch.log(torch.tensor(2.0))
        self.net = nets.MLP(hidden_dim + obs_dim, hidden_dim, 2 * act_space.shape[0], layers_num)

    def forward(self, pred_last, obs, deterministic):
        B, T = pred_last.shape[:2]
        assert pred_last.shape == (B, T, self.hidden_dim), pred_last.shape
        assert obs.shape == (B, T, self.obs_dim), obs.shape

        x = torch.cat([pred_last, obs], dim=-1)
        x = self.net(x)
        mean, log_std = torch.chunk(x, chunks=2, dim=-1)	# 正态分布 均值、标准差对数
        std = torch.exp(torch.clamp(log_std, -20, 2))
        dist = Normal(mean, std)

        u = mean if deterministic else dist.rsample()
        action = F.tanh(u) * self.action_scale + self.action_offset

        # 数值稳定性更好
        log_prob = dist.log_prob(u) - 2 * (self.log2 - u - F.softplus(-2 * u))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, None, log_prob

class ActorDiscrete(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_space, layers_num=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.net = nets.MLP(hidden_dim + obs_dim, hidden_dim, action_space.n, layers_num)

    def forward(self, pred_last, obs, deterministic):
        B, T = pred_last.shape[:2]
        assert pred_last.shape == (B, T, self.hidden_dim), pred_last.shape
        assert obs.shape == (B, T, self.obs_dim), obs.shape

        x = torch.cat([pred_last, obs], dim=-1)
        x = self.net(x)
        prob = F.softmax(x, dim=-1)

        if deterministic:
            action = torch.argmax(prob, dim=-1)
        else:
            action = torch.distributions.Categorical(prob).sample()
        
        log_prob = torch.log(prob + 1e-8)
        return action, prob, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_space, layers_num = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_discrete = tools.is_discrete(act_space)

        if self.act_discrete:
            critic_in_dim = hidden_dim + obs_dim
            critic_out_dim = act_space.n
        else:
            critic_in_dim = hidden_dim + obs_dim + act_space.shape[0]
            critic_out_dim = 1

        self.q1 = nets.MLP(critic_in_dim, hidden_dim, critic_out_dim, layers_num)
        self.q2 = nets.MLP(critic_in_dim, hidden_dim, critic_out_dim, layers_num)
    
    def forward(self, pred_last, obs, action=None):
        B, T = pred_last.shape[:2]
        assert pred_last.shape == (B, T, self.hidden_dim), pred_last.shape
        assert obs.shape == (B, T, self.obs_dim), obs.shape

        if self.act_discrete:
            x = torch.cat([pred_last, obs], dim=-1)
        else:
            assert action is not None
            x = torch.cat([pred_last, obs, action], dim=-1)

        return self.q1(x), self.q2(x)
    
class Agent(nn.Module):
    def __init__(self, obs_space, hidden_dim, act_space, lr, grad_clip, alpha, tau, gamma, loss_scales=None, device=None, use_amp=False):
        super().__init__()
        # ========== 参数保存 ========== 
        self.obs_shape = obs_space.shape
        self.act_space = act_space
        self.hidden_dim = hidden_dim
        self.obs_image = tools.is_image(obs_space)
        self.act_discrete = tools.is_discrete(act_space)
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_scales = loss_scales or {"obs":1.0, "rew":1.0, "con":1.0, "actor":1.0, "critic":1.0, "alpha":1.0}
        self.use_amp = use_amp

        if self.act_discrete:
            self.target_entropy = -0.5 * np.log(1.0 / act_space.n).item()
        else:
            self.target_entropy = -np.prod(act_space.shape).item()
        # ========== encoder ==========
        if self.obs_image:
            channels = obs_space.shape[0]
            self.encoder = nets.ImageEncoder(channels, hidden_dim).to(device)
            obs_dim = hidden_dim
        else:
            obs_dim = math.prod(obs_space.shape)
        
        # ========== predictor, decoder ========== 
        self.predictor = Predictor(obs_dim, act_space, hidden_dim, 3).to(device)
        self.decoder = nets.Decoder(hidden_dim, obs_space).to(device)

        # ========== actor, critic ========== 
        if self.act_discrete:
            self.actor = ActorDiscrete(obs_dim, hidden_dim, act_space).to(device)
        else:
            self.actor = ActorContinuous(obs_dim, hidden_dim, act_space).to(device)
        
        self.critic = Critic(obs_dim, hidden_dim, act_space, 3).to(device)
        self.target_critic = Critic(obs_dim, hidden_dim, act_space, 3).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        for param in self.target_critic.parameters():
            param.requires_grad = False

        # ========== alpha ========== 
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha), dtype=torch.float, device=device))

        # ========== optimizer ========== 
        self.optim = torch.optim.Adam(self.parameters(), lr)
        self.scaler = torch.GradScaler(device, enabled=use_amp)

        # ========== save loss for report ========== 
        self.losses = {}
        self.metrics = {}

    # 初始状态（初始carry）
    def initial(self, batch_size=1):
        pred_last = torch.zeros((batch_size, 1, self.hidden_dim), device=self.device)
        hx = None
        return pred_last, hx
    
    def take_action(self, obs, carry, deterministic):
        self.eval()
        with torch.no_grad():
            pred_last, hx = carry
            obs = torch.tensor(obs, dtype=torch.float, device=self.device).reshape((1, 1, *self.obs_shape))
            feat = self.encoder(obs) if self.obs_image else obs
            action, _, _ = self.actor(pred_last, feat, deterministic)
            pred, hx = self.predictor(feat, action, hx)
        action = action.cpu().numpy().reshape(self.act_space.shape)
        if tools.is_discrete(self.act_space):
            action = action.item()
        return action, (pred, hx)
    
    def train_step(self, obs, reward, con, action):
        # 当report=False时，正常训练，不计算metrics
        # 当report=True时，计算metrics，但不更新模型
        B, T = obs.shape[:2]
        assert obs.shape == (B, T, *self.obs_shape), obs.shape
        assert reward.shape == (B, T, 1), reward.shape
        assert con.shape == (B, T, 1), con.shape
        assert action.shape == (B, T, *self.act_space.shape), action.shape

        self.losses = {}
        self.train()
        
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.use_amp):
            # 压缩绝对值过大的奖励
            reward = tools.symlog(reward)
            feat = self.encoder(obs) if self.obs_image else obs
            pred, hx = self.predictor(feat, action)
            pred_last = torch.cat([self.initial(B)[0], pred[:, :-1]], dim=1)

            # ========== predict loss ========== 
            # o_{t+1} = f(h_t)
            pred_obs, pred_rew, pred_con, pred_exceed = self.decoder(pred[:, :-1])
            if self.obs_image:
                self.losses["obs"] = self.weighted_image_loss(pred_obs, obs[:, 1:], obs[:, :-1])
            else:
                self.losses["obs"] = F.mse_loss(pred_obs, obs[:, 1:])
            self.losses["rew"] = F.mse_loss(pred_rew, reward[:, 1:])
            self.losses["con"] = F.binary_cross_entropy_with_logits(pred_con, con[:, 1:])

            # exceed, length = self.reward_exceeds_mean(reward, 10)
            # self.losses["exceed"] = F.binary_cross_entropy_with_logits(pred_exceed[:, :length], exceed)

        # ========== actor critic loss ========== 
        if self.act_discrete:
            td_target = self.td_target_discrete(pred, feat, reward, con)
            self.losses["critic"] = self.critic_loss_discrete(pred_last, feat, action, td_target)
            self.losses["actor"], entropy = self.actor_loss_discrete(pred_last, feat)
        else:
            td_target = self.td_target_continuous(pred, feat, reward, con)
            self.losses["critic"] = self.critic_loss_continuous(pred_last, feat, action, td_target)
            self.losses["actor"], entropy = self.actor_loss_continuous(pred_last, feat)

        # ========== alpha loss ==========
        self.losses["alpha"] = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())

        # ========== total loss ==========
        self.losses["total"] = sum(self.loss_scales[key] * value for key,value in self.losses.items())

        # ========== update==========
        self.optim.zero_grad()
        self.scaler.scale(self.losses["total"]).backward()
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=self.grad_clip, norm_type=2)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.grad_clip, norm_type=2)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip, norm_type=2)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip, norm_type=2)
        self.scaler.step(self.optim)
        self.scaler.update()
        self.soft_update(self.critic, self.target_critic)

        # 这里不使用.mean().item()，在report中再使用，减小开销
        self.metrics["entropy_mean"] = entropy
        self.metrics["target_mean"] = td_target

    def report(self):
        metrics = {}
        for name, value in self.losses.items():
            metrics["loss/" + name] = value.item()
        if self.metrics:
            metrics["entropy_mean"] = self.metrics["entropy_mean"].mean().item()
            metrics["target_mean"] = self.metrics["target_mean"].mean().item()
        metrics["alpha"] = self.log_alpha.exp().item()
        metrics["grad_norm/predictor"] = tools.get_grad_norm(self.predictor)
        metrics["grad_norm/decoder"] = tools.get_grad_norm(self.decoder)
        metrics["grad_norm/actor"] = tools.get_grad_norm(self.actor)
        metrics["grad_norm/critic"] = tools.get_grad_norm(self.critic)
        metrics["grad_norm/total"] = tools.get_grad_norm(self)
        return metrics
    
    def video_pred(self, obs, reward, con, action):
        assert self.obs_image
        obs = obs[:6]
        action = action[:6]

        with torch.no_grad():
            feat = self.encoder(obs) if self.obs_image else obs
            pred, hx = self.predictor(feat, action)
            pred_obs, pred_rew, pred_con, _ = self.decoder(pred[:, :-1])
            real_obs = obs[:, 1:]
            error = (pred_obs - real_obs + 1.0) / 2.0
            video = torch.cat([real_obs, pred_obs, error], 3)		# (B, T, C, H, W) 在高度方向拼接

            if video.shape[-3] == 1:
                video = video.repeat(1, 1, 3, 1, 1)

            video = video.cpu().numpy()
        B, T, C, H, W = video.shape
        video = video.transpose(1, 2, 3, 0, 4).reshape((1, T, C, H, B * W))
        return video

    def td_target_continuous(self, pred, feat, reward, con):
        with torch.no_grad():
            action_next, _, log_prob = self.actor(pred[:, :-1], feat[:, 1:], False)
            entropy = -log_prob
            q1, q2 = self.target_critic(pred[:, :-1], feat[:, 1:], action_next)
            next_value = torch.min(q1, q2) + self.log_alpha.exp() * entropy
            td_target = reward[:, 1:] + self.gamma * next_value * con[:, 1:]
        return td_target
    
    def td_target_discrete(self, pred, feat, reward, con):
        with torch.no_grad():
            action_next, prob, log_prob = self.actor(pred[:, :-1], feat[:, 1:], False)
            entropy = -torch.sum(prob * log_prob, dim=-1, keepdim=True)
            q1, q2 = self.target_critic(pred[:, :-1], feat[:, 1:])
            q_min = torch.sum(prob * torch.min(q1, q2), dim=-1, keepdim=True)
            next_value = q_min + self.log_alpha.exp() * entropy
            td_target = reward[:, 1:] + self.gamma * next_value * con[:, 1:]
        return td_target
    
    def critic_loss_continuous(self, pred_last, feat, action, td_target):
        # 需要detach吗？
        pred_last = pred_last[:, :-1].detach()
        feat = feat[:, :-1].detach()
        # feat = tools.grad_scale(feat[:, :-1], 0.1)
        action = action[:, :-1].detach()

        q1, q2 = self.critic(pred_last, feat, action)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        return critic_loss
    
    def critic_loss_discrete(self, pred_last, feat, action, td_target):
        # 需要detach吗？
        pred_last = pred_last[:, :-1].detach()
        feat = feat[:, :-1].detach()
        # feat = tools.grad_scale(feat[:, :-1], 0.1)

        q1, q2 = self.critic(pred_last, feat)
        q1 = q1.gather(-1, action[:, :-1].unsqueeze(-1))
        q2 = q2.gather(-1, action[:, :-1].unsqueeze(-1))
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        return critic_loss

    def actor_loss_continuous(self, pred_last, feat):
        self.freeze_params(self.critic.parameters(), True)
        #丢弃最后一个时间步，防止遇到终止情况
        pred_last = pred_last[:, :-1].detach()
        feat = feat[:, :-1].detach()
        action_new, _, log_prob = self.actor(pred_last, feat, False)
        entropy = -log_prob
        q1, q2 = self.critic(pred_last, feat, action_new)
        actor_loss = torch.mean(-self.log_alpha.detach().exp() * entropy - torch.min(q1, q2))
        self.freeze_params(self.critic.parameters(), False)
        return actor_loss, entropy
    
    def actor_loss_discrete(self, pred_last, feat):
        self.freeze_params(self.critic.parameters(), True)
        #丢弃最后一个时间步，防止遇到终止情况
        pred_last = pred_last[:, :-1].detach()
        feat = feat[:, :-1].detach()
        action_new, prob, log_prob = self.actor(pred_last, feat, False)
        entropy = -torch.sum(prob * log_prob, dim=-1, keepdim=True)
        q1, q2 = self.critic(pred_last, feat)
        q_min = torch.sum(prob * torch.min(q1, q2), dim=-1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.detach().exp() * entropy - q_min)
        self.freeze_params(self.critic.parameters(), False)
        return actor_loss, entropy
    
    def soft_update(self, net, target_net):
        for p_target, p in zip(target_net.parameters(), net.parameters()):
            p_target.data.copy_(p_target.data * (1.0 - self.tau) + p.data * self.tau)

    def freeze_params(self, params_to_freeze, freeze):
        for p in params_to_freeze:
            p.requires_grad = not freeze

    def reward_exceeds_mean(self, reward, window_size):
        with torch.no_grad():
            B, T = reward.shape[:2]
            reward = reward.reshape((B, T))
            # 去除第一个时间步，因为这里需要用t-1时刻信息预测t~(t+k-1)时刻
            reward = reward[:, 1:]
            # 计算每个batch的平均奖励
            reward_mean = reward.mean(dim=1).reshape(B, 1, 1)
            # 在长度为window_size的滑动窗口内统计是否有超过平均奖励的值
            reward_unfold = reward.unfold(dimension=1, size=window_size, step=1)
            exceeds = (reward_unfold > reward_mean).any(dim=2).float()
            T = exceeds.shape[1]
        return exceeds.reshape((B, T, 1)), T
    
    def weighted_image_loss(self, pred_obs, real_obs, last_obs):
        assert len(pred_obs.shape) == 5, pred_obs.shape
        B, T, C, H, W = pred_obs.shape
        pred_weight = torch.square(pred_obs - real_obs)
        diff_weight = torch.square(real_obs - last_obs)
        weight = pred_weight + diff_weight
        # 让权重之和为1
        weight = F.softmax(weight.reshape(B, T, C, H*W), dim=-1).reshape(B, T, C, H, W)
        loss = weight * torch.square(pred_obs - real_obs)
        return loss.sum()

