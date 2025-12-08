import torch
from train import UniFracEncoder, distance_matching_loss_batch

B,R,L = 2, 4, 150
seq_ids = torch.randint(0,5,(B,R,L))
model = UniFracEncoder()
emb = model(seq_ids)
print("emb shape:", emb.shape)