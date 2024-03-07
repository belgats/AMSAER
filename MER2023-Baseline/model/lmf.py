"""
paper: Efficient Low-rank Multimodal Fusion with Modality-Specific Factors
From: https://github.com/Justin1904/Low-rank-Multimodal-Fusion
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from .encoders import MLPEncoder, LSTMEncoder

class LMF(nn.Module):

    def __init__(self, text_dim,audio_dim, video_dim, output_dim1, output_dim2, output_dim3, rank, dropout, hidden_dim, grad_clip, feat_type ):
        super(LMF, self).__init__()

        # load input and output dim
        self.text_dim    = text_dim
        self.audio_dim   = audio_dim
        self.video_dim   = video_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.output_dim3 = output_dim3
        self.rank        = rank
        self.dropout     = dropout
        self.hidden_dim  = hidden_dim
        self.grad_clip   = grad_clip
        self.feat_type   = feat_type 
        # define the pre-fusion subnetworks
        if self.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(self.audio_dim, self.hidden_dim, self.dropout)
            self.text_encoder  = MLPEncoder(self.text_dim,  self.hidden_dim, self.dropout)
            self.video_encoder = MLPEncoder(self.video_dim, self.hidden_dim, self.dropout)
        elif self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(self.audio_dim, self.hidden_dim, self.dropout)
            self.text_encoder  = LSTMEncoder(self.text_dim,  self.hidden_dim, self.dropout)
            self.video_encoder = LSTMEncoder(self.video_dim, self.hidden_dim, self.dropout)

        # define the post_fusion layers
        self.output_dim = hidden_dim // 2
        self.post_fusion_dropout = nn.Dropout(p=self.dropout)
        self.audio_factor   = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.video_factor   = Parameter(torch.Tensor(self.rank, hidden_dim + 1, self.output_dim))
        self.text_factor    = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias    = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        self.fc_out_1 = nn.Linear(self.output_dim, self.output_dim1)
        self.fc_out_2 = nn.Linear(self.output_dim, self.output_dim2)
        self.fc_out_3 = nn.Linear(self.output_dim, self.output_dim3)


    def forward(self, batch):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x:  tensor of shape (batch_size, text_in)
        '''
        audio_h = self.audio_encoder(batch['audios'])
        video_h = self.video_encoder(batch['videos'])
        text_h  = self.text_encoder(batch['texts'])
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one  = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(audio_h.device)
        _audio_h = torch.cat((add_one, audio_h), dim=1)
        _video_h = torch.cat((add_one, video_h), dim=1)
        _text_h  = torch.cat((add_one, text_h), dim=1)
        
        # torch.matmul() 处理时会将 [batch, feat+1] -> [rank, batch, feat+1], 看结果就好像把 [feat+1] 分解为 rank * [hidden]
        fusion_audio = torch.matmul(_audio_h, self.audio_factor) # [batch, feat+1] * [rank, feat+1, hidden] = [rank, batch, hidden]
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text  = torch.matmul(_text_h,  self.text_factor )
        fusion_zy    = fusion_audio * fusion_video * fusion_text # [rank, batch, hidden]

        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias # [1, rank] * [batch, rank, hidden] -> [batch, hidden]
        features = output.view(-1, self.output_dim)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        sents_out  = self.fc_out_3(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, sents_out, interloss
