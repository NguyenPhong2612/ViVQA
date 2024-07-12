import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available else "cpu")
vision_model_name = "google/vit-base-patch16-224-in21k"
language_model_name = "vinai/phobert-base"



def generate_padding_mask(sequences, padding_idx):
    if sequences is None:
        return None
    if len(sequences.shape) == 2:
        __seq = sequences.unsqueeze(dim=-1)
    else:
        __seq = sequences
    mask = (torch.sum(__seq, dim=-1) == (padding_idx*__seq.shape[-1])).long() * -10e4
    return mask.unsqueeze(1).unsqueeze(1)


class ScaledDotProduct(nn.Module):
    def __init__(self, d_model = 512, h = 8, d_k = 64, d_v = 64):
        super().__init__()

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            att += attention_mask
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, att


class MultiheadAttention(nn.Module):
    
    def __init__(self, d_model = 512, dropout = 0.1, use_aoa = True):
        super().__init__()
        self.d_model = d_model
        self.use_aoa = use_aoa
        
        self.attention = ScaledDotProduct()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if self.use_aoa:
            self.infomative_attention = nn.Linear(2 * self.d_model, self.d_model)
            self.gated_attention = nn.Linear(2 * self.d_model, self.d_model)
        
    def forward(self, q, k, v, mask = None):
        out, _  = self.attention(q, k, v, mask)
        if self.use_aoa:
            aoa_input = torch.cat([q, out], dim = -1)
            i = self.infomative_attention(aoa_input)
            g = torch.sigmoid(self.gated_attention(aoa_input))
            out = i * g
        return out
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        out = self.fc1(input)
        out = self.fc2(self.relu(out))
        return out
    
class AddNorm(nn.Module):
    def __init__(self, dim = 512, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, y):
        return self.norm(x + self.dropout(y))
   
    
class SinusoidPositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats=512, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            embed = embed / (embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=-1).flatten(-2)

        return pos


class GuidedEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_mhatt = MultiheadAttention()
        self.guided_mhatt = MultiheadAttention()
        self.pwff = PositionWiseFeedForward()
        self.first_norm = AddNorm()
        self.second_norm = AddNorm()
        self.third_norm = AddNorm()
    def forward(self, q, k, v, self_mask, guided_mask):
        self_att = self.self_mhatt(q, q, q, self_mask)
        self_att = self.first_norm(self_att, q)
        guided_att = self.guided_mhatt(self_att, k, v, guided_mask)
        guided_att = self.second_norm(guided_att, self_att)
        out = self.pwff(guided_att)
        out = self.third_norm(out, guided_att)
        return out


class GuidedAttentionEncoder(nn.Module):
    def __init__(self, num_layers = 2, d_model = 512):
        super().__init__()
        self.pos_embedding = SinusoidPositionalEmbedding()
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.guided_layers = nn.ModuleList([GuidedEncoderLayer() for _ in range(num_layers)])
        self.language_layers = nn.ModuleList(GuidedEncoderLayer() for _ in range(num_layers))
    
    def forward(self, vision_features, vision_mask, language_features, language_mask):
        vision_features = self.layer_norm(vision_features) + self.pos_embedding(vision_features)
        language_features = self.layer_norm(language_features) + self.pos_embedding(language_features)
        
        for layers in zip(self.guided_layers, self.language_layers):
            guided_layer, language_layer = layers
            vision_features = guided_layer(q = vision_features,
                                          k = language_features,
                                          v = language_features,
                                          self_mask = vision_mask,
                                          guided_mask = language_mask)
            language_features = language_layer(q = language_features,
                                              k = vision_features,
                                              v = vision_features,
                                              self_mask = language_mask,
                                              guided_mask = vision_mask)
            
            return vision_features, language_features


class VisionEmbedding(nn.Module):
    def __init__(self, out_dim = 768, hidden_dim = 512, dropout = 0.1):
        super().__init__()
        self.prep = AutoFeatureExtractor.from_pretrained(vision_model_name)
        self.backbone = AutoModel.from_pretrained(vision_model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.proj = nn.Linear(out_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    def forward(self, images):
        inputs = self.prep(images = images, return_tensors = "pt").to(device)
        with torch.no_grad():
            outputs = self.backbone(**inputs)
        features = outputs.last_hidden_state
        vision_mask = generate_padding_mask(features, padding_idx = 0)
        out = self.proj(features)
        out = self.gelu(out)
        out = self.dropout(out)
        return out, vision_mask
    

class LanguageEmbedding(nn.Module):
    def __init__(self, out_dim = 768, hidden_dim = 512, dropout = 0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.embeding = AutoModel.from_pretrained(language_model_name)
        for param in self.embeding.parameters():
            param.requires_grad = False
        self.proj = nn.Linear(out_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    def forward(self, questions):
        inputs = self.tokenizer(questions,
                                padding = 'max_length',
                                max_length = 30,
                                truncation = True,
                                return_tensors = 'pt',
                                return_token_type_ids = True,
                                return_attention_mask = True).to(device)
        
        features = self.embeding(**inputs).last_hidden_state
        language_mask = generate_padding_mask(inputs.input_ids, padding_idx=self.tokenizer.pad_token_id)
        out = self.proj(features)
        out = self.gelu(out)
        out = self.dropout(out)
        return out, language_mask

class BaseModel(nn.Module):
    def __init__(self, num_classes = 353, d_model = 512):
        super().__init__()
        self.vision_embedding = VisionEmbedding()
        self.language_embedding = LanguageEmbedding()
        self.encoder = GuidedAttentionEncoder()
        self.fusion = nn.Sequential(nn.Linear(2 * d_model, d_model),
                                  nn.ReLU(),
                                  nn.Dropout(0.2))
        self.classify = nn.Linear(d_model, num_classes)
        self.attention_weights = nn.Linear(d_model, 1)
    
    def forward(self, images, questions):
        embedded_text, text_mask = self.language_embedding(questions)
        embedded_vision, vison_mask = self.vision_embedding(images)

        encoded_image, encoded_text = self.encoder(embedded_vision, vison_mask,embedded_text, text_mask)
        text_attended = self.attention_weights(torch.tanh(encoded_text))
        image_attended = self.attention_weights(torch.tanh(encoded_image))
        
        attention_weights = torch.softmax(torch.cat([text_attended, image_attended], dim=1), dim=1)
        
        attended_text = torch.sum(attention_weights[:, 0].unsqueeze(-1) * encoded_text, dim=1)
        attended_image = torch.sum(attention_weights[:, 1].unsqueeze(-1) * encoded_image, dim=1)
        
        fused_output = self.fusion(torch.cat([attended_text, attended_image], dim=1))
        logits = self.classify(fused_output)
        logits = F.log_softmax(logits, dim=-1)
        return logits
    


if __name__ == "__main__":
    model = BaseModel().to(device)
    print(model.eval)
        