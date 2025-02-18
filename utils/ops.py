import torch

def get_image_understanding(embeds):
    query_vectors = embeds[0,1:,:] # S X D
    norm_query = torch.linalg.norm(query_vectors, dim=-1) # S
    query_vectors = query_vectors.t() # D X S
    _query_vectors = query_vectors.unsqueeze(1)
    query_vectors = query_vectors.unsqueeze(-1)
    sim_map = torch.matmul(query_vectors, _query_vectors) # D X S X S
    sim_map = torch.sum(sim_map, dim=0)
    norm_query = torch.matmul(norm_query.unsqueeze(0), norm_query.unsqueeze(-1))
    norm_query = torch.clamp(norm_query, min=1e-8)
    sim_map = sim_map / norm_query # S X S
    
    return sim_map