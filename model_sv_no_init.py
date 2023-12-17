import torch
import torchvision.io as io
import json
import wandb
import os
import glob
from transformers import CLIPModel
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize, FSQ



# from utils import si

class SelfAttn(nn.Module):
    def __init__(self, ):
        pass
    def forward(self, x):
        # x : b * 
        
        pass

class DecodeNet(nn.Module):
    def __init__(self, indim, ):
        pass
    def forward(self, x):
        pass

class SkillAutoEncoder(nn.Module):
    def __init__(self, conf):
        super(SkillAutoEncoder, self).__init__()

        self.action_dim = conf['act_dim']
        self.act_emb = conf['act_emb']
        self.obs_emb = conf['obs_emb']
        self.input_emb = conf['input_emb']
        self.encoder_layer_num = conf['encoder_layer_num']
        self.encoder_dim = conf['encoder_dim']
        self.num_head = conf['num_head']
        self.codebook_dim = conf['codebook_dim']
        self.codebook_entry = conf['codebook_entry']
        self.skill_block_size = conf['skill_block_size']
        self.using_emb = conf['using_emb']
        self.vq_type = conf['vq_type']
        self.vq_decay = conf['vq_decay']
        self.commitment_loss_weight = conf['commitment_loss_weight']
        self.fsq_level = conf['fsq_level']
        self.decoder_type = conf['decoder_type']
        self.gru_hidden_dim = conf['gru_hidden_dim']
        self.obs_red_emb = conf['obs_red_emb']
             
        # define modules
        self.obs_mlp = nn.Linear(self.obs_emb,self.obs_red_emb)
        self.obs_action_mlp = nn.Linear(self.action_dim + self.obs_red_emb, self.input_emb)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_emb, nhead=self.num_head, batch_first=True)
        self.encoder =  nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_layer_num)
     
        if self.vq_type == 'vq-vae':
            self.prediction_head = nn.Linear(self.input_emb * self.skill_block_size, self.codebook_dim)
            self.vq = VectorQuantize(
                            dim = self.codebook_dim,
                            codebook_size = self.codebook_entry,     # codebook size
                            decay = self.vq_decay,             # the exponential moving average decay, lower means the dictionary will change faster
                            commitment_weight = self.commitment_loss_weight   # the weight on the commitment loss
                        )
            # self.vq.codebook = self.vq.codebook * 0. 
        
        elif self.vq_type == 'fsq':
            self.prediction_head = nn.Linear(self.input_emb, len(self.fsq_level))
            self.vq = FSQ(self.fsq_level)
            self.lift = nn.Linear(len(self.fsq_level)+self.obs_red_emb, self.gru_hidden_dim)
            self.lift_wo_init = nn.Linear(len(self.fsq_level), self.gru_hidden_dim)
            # lift_layers = [
            #     nn.Linear(len(self.fsq_level), self.codebook_dim * 4),
            #     nn.ReLU(),
            #     nn.Linear(self.codebook_dim * 4, self.codebook_dim * 2),
            #     nn.ReLU(),
            #     nn.Linear(self.codebook_dim * 2, self.codebook_dim * 2),
            #     nn.ReLU(),
            #     nn.Linear(self.codebook_dim * 2, self.codebook_dim),
            # ]
            # self.lift = nn.Sequential(*lift_layers)
        else:
            raise NotImplementedError('Unknown vq_type')
        
        
        if self.decoder_type == 'mlp':
        
            d_layers=[
                            nn.Linear(self.codebook_dim + self.obs_red_emb, self.input_emb * 2),
                            nn.ReLU(),
                            nn.Linear(self.input_emb * 2, self.input_emb * 4),
                            nn.ReLU(),
                            nn.Linear(self.input_emb * 4, self.input_emb * 4),
                            nn.ReLU(),
                            nn.Linear(self.input_emb * 4, self.input_emb * 8),
                            nn.ReLU(),
                            nn.Linear(self.input_emb * 8, self.skill_block_size * self.action_dim)
            ]
            self.skill_decoder = nn.Sequential(*d_layers)
        
        elif self.decoder_type == 'transformer':
            raise NotImplementedError('transformer not implemented')

        elif self.decoder_type == 'rnn':
            self.gru_unit_1 = nn.GRU(input_size= self.gru_hidden_dim, hidden_size=self.gru_hidden_dim, batch_first=True)
            self.gru_unit_2 = nn.GRU(input_size=self.gru_hidden_dim, hidden_size=self.gru_hidden_dim, batch_first=True)
            self.gru_unit_3 = nn.GRU(input_size=self.gru_hidden_dim, hidden_size=self.gru_hidden_dim//4, batch_first=True)
            self.relu = nn.ReLU()
            self.hidden_to_action = nn.Linear(self.gru_hidden_dim, self.action_dim)
        else:
            raise NotImplementedError('Unknown decoder_type')

        if not self.using_emb:
        # load encoder for obseravtion side
            if conf['image_encoder'] == 'clip':
                class clip_vision_wrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    def forward(self, x):
                        return self.model.get_image_features(x)
                
                self.image_encoder = clip_vision_wrapper(
                                        CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                                    )
                if conf['freeze_image_encoder']:
                    for param in self.image_encoder.parameters():
                        param.requires_grad = False
                else:
                    pass
            else:
                raise NotImplementedError

    def encode(self, obs, act):
        # obs: B * T * 3 * 224 * 224
        # act: B * T * 2
        # obs_embed: B * T * 512(clip)
        # act_embed: B * T * act_emb
        # out: B * T * input_emb
        if not self.using_emb:
            B, T, C, H, W = obs.shape
            obs = obs.view(B*T, C, H, W)
            obs_embed = self.image_encoder(obs)
            obs_embed = obs_embed.view(B, T, -1)
        else:
            obs_embed = obs
        # print(obs_embed.shape)
        obs_reduced_emb = self.obs_mlp(obs_embed)
        # act_embed = self.action_mlp(act)
        # print(act_embed.shape)

        input_embed = torch.cat([obs_reduced_emb, act], -1) # B * T * (obs+act)
        input_embed = self.obs_action_mlp(input_embed)
        out = self.encoder(input_embed)
        
        return out   

    def decode(self, x, init=None):
        B, T, C = x.shape
        # x = x.view(B, T*C)      
        out = self.prediction_head(x) # B * codebook_dim (4 for fsq)


        # z, vq_loss, pp = self.codebook_search(self.codebook, out)
        #out -> z
        if self.vq_type == 'vq-vae':
            z, indices, commitment_loss = self.vq(out)
            vq_loss = commitment_loss 
            pp = torch.unique(indices).shape[0] / self.vq.codebook_size
        elif self.vq_type == 'fsq':
            z, indices = self.vq(out)
            # z = self.lift(z)
            vq_loss = torch.tensor(0.)
            # print(indices.shape, 'indices_shape')
            # print(torch.unique(indices).shape, 'unique_indices')
            pp = torch.unique(indices).shape[0] / self.vq.n_codes

        if init is not None:
            if not self.using_emb:
                init = self.image_encoder(init)
            init = self.obs_mlp(init)
                # print(init.shape, 'init_shape')
            init = init.unsqueeze(1)
            init = init.repeat(1, T, 1)
                # print(init.shape, 'init_shape_2')
            # print(init.shape, 'init_shape')
            z_o = torch.cat([z, init], -1)
        # print(z.shape, 'z_shape')
        # z_o = self.lift(z_o)
        z_o = self.lift_wo_init(z)
        # print(z_o.shape, 'z_o_shape')
        if self.decoder_type == 'mlp':
            out = self.skill_decoder(z_o)
        elif self.decoder_type == 'rnn':
            out,_ = self.gru_unit_1(z_o)
            out = self.relu(out)
            out,_ = self.gru_unit_2(out)
            out = self.relu(out)
            # out,_ = self.gru_unit_3(out)
            # # print(out.shape, 'out_shape')
            # out = self.relu(out)
            out = self.hidden_to_action(out)
        out = out.view(B, T, self.action_dim)
 
        return out, vq_loss, pp

    def decode_eval(self, x, init=None):
        B, T, C = x.shape
        z_o = self.lift_wo_init(x)
        # print(z_o.shape, 'z_o_shape')
        if self.decoder_type == 'mlp':
            out = self.skill_decoder(z_o)
        elif self.decoder_type == 'rnn':
            out,_ = self.gru_unit_1(z_o)
            out = self.relu(out)
            out,_ = self.gru_unit_2(out)
            out = self.relu(out)
            # out,_ = self.gru_unit_3(out)
            # # print(out.shape, 'out_shape')
            # out = self.relu(out)
            out = self.hidden_to_action(out)
        out = out.view(B, T, self.action_dim)
        return out

    def get_decode_loss(self, ):
        
        pass
    
    def get_codebook_KLloss(self, ):
        
        pass
    
    def forward(self, obs, act):
        # gen_output and loss
        out = self.encode(obs, act)
        out, vq_loss, pp = self.decode(out, init=None)
        return out, vq_loss, pp

if __name__ == '__main__':


    conf = {
            "act_dim": 8,
            "act_emb": 8,
            "obs_emb": 512,
            "input_emb": 2048,
            "encoder_layer_num": 2,
            "encoder_dim": None,
            "num_head": 8,
            "codebook_dim": 32,
            "codebook_entry": 1024,
            "skill_block_size": 32,
            "image_encoder":'clip',
            "freeze_image_encoder": 1,
            "using_emb":True,
            "vq_type": 'fsq',
            "vq_decay": 0.8,
            "commitment_loss_weight": 1,
            "fsq_level": [8, 5, 5, 5],
            "decoder_type": 'rnn',
            "gru_hidden_dim": 2048,
            "obs_red_emb": 64
            }
    
    net = SkillAutoEncoder(conf)
    
    obs = torch.rand(4, 32, 512)
    act = torch.rand(4, 32, 8)

    out = net.encode(obs, act)
    print(out.shape)
    out, vq_loss, pp = net.decode(out, init=None)
    

    print(out.shape, vq_loss, pp)