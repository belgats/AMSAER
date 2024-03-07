"""
paper: Tensor Fusion Network for Multimodal Sentiment Analysis
From: https://github.com/A2Zadeh/TensorFusionNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from  .encoders import MLPEncoder, LSTMEncoder

class TFN(nn.Module):

    def __init__(self,  audio_dim, text_dim, video_dim, output_dim1, output_dim2, output_dim3, dropout, hidden_dim, grad_clip,feat_type):

        super(TFN, self).__init__()

        self.text_dim    = text_dim
        self.audio_dim   = audio_dim
        self.video_dim   = video_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.output_dim3 = output_dim3
        self.dropout     = dropout
        self.hidden_dim  = hidden_dim
        self.grad_clip   = grad_clip
        self.feat_type   = feat_type
        # define the pre-fusion subnetworks [感觉输入的audio/video是句子级别，但是 text是词级别信息]
        if self.feat_type in ['utt']:
            self.audio_encoder = MLPEncoder(self.audio_dim, self.hidden_dim, self.dropout)
            self.text_encoder  = MLPEncoder(self.text_dim,  self.hidden_dim, self.dropout)
            self.video_encoder = MLPEncoder(self.video_dim, self.hidden_dim, self.dropout)
        elif self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(self.audio_dim, self.hidden_dim, self.dropout)
            self.text_encoder  = LSTMEncoder(self.text_dim,  self.hidden_dim, self.dropout)
            self.video_encoder = LSTMEncoder(self.video_dim, self.hidden_dim, self.dropout)
        
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.dropout)
        self.post_fusion_layer_1 = nn.Linear((self.hidden_dim + 1) * (self.hidden_dim + 1) * (self.hidden_dim + 1), self.hidden_dim)
        self.post_fusion_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc_out_1 = nn.Linear(self.hidden_dim, self.output_dim1)
        self.fc_out_2 = nn.Linear(self.hidden_dim, self.output_dim2)
        self.fc_out_3 = nn.Linear(self.hidden_dim, self.output_dim3)


    # audio/video是句子级别, text的word level
    def forward(self, batch):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_dim)
            video_x: tensor of shape (batch_size, video_dim)
            text_x:  tensor of shape (batch_size, text_dim )
        '''

        audio_h = self.audio_encoder(batch['audios'])
        text_h  = self.text_encoder(batch['texts'])
        video_h = self.video_encoder(batch['videos'])
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one  = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(audio_h.device)
        _audio_h = torch.cat((add_one, audio_h), dim=1)
        _video_h = torch.cat((add_one, video_h), dim=1)
        _text_h  = torch.cat((add_one, text_h), dim=1)

        # outer product
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.hidden_dim + 1) * (self.hidden_dim + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        features = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        sents_out  = self.fc_out_3(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, sents_out, interloss
