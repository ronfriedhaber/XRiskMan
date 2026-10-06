"""Causal numerical transformer policy and rolling history."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from config import ModelConfig

class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); self.heads=c.heads
        self.norm1,self.norm2=nn.LayerNorm(c.width),nn.LayerNorm(c.width)
        self.qkv,self.proj=nn.Linear(c.width,3*c.width),nn.Linear(c.width,c.width)
        self.mlp=nn.Sequential(nn.Linear(c.width,4*c.width),nn.GELU(),nn.Linear(4*c.width,c.width))
    def forward(self,x):
        b,t,w=x.shape; q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,w//self.heads).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
        x=x+self.proj(a.transpose(1,2).reshape(b,t,w)); return x+self.mlp(self.norm2(x))

class Policy(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__(); self.config=config
        self.embed=nn.Linear(config.input_dim,config.width); self.position=nn.Parameter(torch.zeros(1,config.context,config.width))
        self.blocks=nn.Sequential(*(Block(config) for _ in range(config.layers))); self.norm=nn.LayerNorm(config.width)
        self.actor,self.critic=nn.Linear(config.width,4),nn.Linear(config.width,1)
        nn.init.normal_(self.position,std=.02); nn.init.normal_(self.actor.weight,std=.01); nn.init.zeros_(self.actor.bias)
    def forward(self,tokens,lengths):
        x=self.blocks(self.embed(tokens)+self.position[:,:tokens.shape[1]])
        x=self.norm(x[torch.arange(len(x),device=x.device),lengths-1]); return Categorical(logits=self.actor(x)),self.critic(x).squeeze(-1)

class History:
    def __init__(self,observation,context,device):
        self.obs_dim=observation.shape[-1]; width=23 if self.obs_dim==18 else 16
        self.tokens=torch.zeros(len(observation),context,width,device=device); self.tokens[:,:1,:self.obs_dim]=torch.as_tensor(observation,device=device)
        self.lengths=torch.ones(len(observation),dtype=torch.long,device=device)
    def push(self,observation,action,reward,reset):
        device,context=self.tokens.device,self.tokens.shape[1]; full=self.lengths==context; self.tokens[full]=self.tokens[full].roll(-1,dims=1)
        self.lengths=(self.lengths+1).clamp(max=context); token=torch.as_tensor(observation,device=device)
        if self.obs_dim==18: token=torch.cat((token,F.one_hot(action,4).float(),torch.as_tensor(reward,device=device).float()[:,None]),dim=-1)
        self.tokens[torch.arange(len(token),device=device),self.lengths-1]=token; reset=torch.as_tensor(reset,device=device); self.tokens[reset]=0
        self.tokens[reset,0,:self.obs_dim]=torch.as_tensor(observation,device=device)[reset]; self.lengths[reset]=1
