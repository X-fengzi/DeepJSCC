import torch
import torch.nn as nn
import torch.nn.functional as F

from model import nets


#--------------------------------------------------  Random Coding -------------------------------------------------------  


########################### Discrete Category Distribution #####################

class DiscreteEncoder(nn.Module):
    def __init__(self, dims= [], net='ToyNet',num_embeddings=4,dim_dic=2):
        super(DiscreteEncoder, self).__init__()
        self.net = getattr(nets, net)(dims)
        self.input_dim, self.output_dim = dims
        self.dim_dic = dim_dic
        self.num_embeddings = num_embeddings

        assert self.output_dim % self.dim_dic == 0, "output_dim must be divisible by dim_dic"

        self.embedding = nn.Parameter(torch.Tensor(self.num_embeddings, self.dim_dic)) ## can be seen as codebooks
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)

    def forward(self, X):
        X = self.net(X)
        X = X.view(-1,self.output_dim//self.dim_dic,self.dim_dic)
        score = torch.matmul(X, self.embedding.transpose(1,0))/torch.sqrt(torch.tensor(self.dim_dic))
        dist = torch.distributions.categorical.Categorical(logits=score)
        # dist = F.softmax(score,dim=-1)
        if self.training:
            samples = F.gumbel_softmax(score,tau=0.5,hard=True)
        else:
            samples = torch.argmax(score,dim=-1)
            samples = F.one_hot(samples,num_classes=self.num_embeddings)
            samples = samples.float()
       
        return samples, self.embedding, dist

########################### Gaussian Distribution #####################

class GaussianEncoder(nn.Module):
    def __init__(self, dims= [], net='ToyNet'):
        super(GaussianEncoder, self).__init__()
        self.input_dim, self.output_dim = dims
        self.net = getattr(nets, net)(dims)
    
    def forward(self, X):
        X = self.net(X) # statistics
        mu = X[:,:self.output_dim//2]
        std = F.softplus(X[:,self.output_dim//2:]-5,beta=1) + 1e-6
        pd = torch.distributions.Normal(mu, std)
        z = pd.rsample()
        return z, pd
    
#--------------------------------------------------  deterministic Coding ------------------------------------------------------- 


class DeterministicEncoder(nn.Module):
    def __init__(self, dims= [], net='ToyNet'):
        super(DeterministicEncoder, self).__init__()
        self.input_dim, self.output_dim = dims
        self.net = getattr(nets, net)(dims)
    
    def forward(self, x):
        z = self.net(x) # statistics
        return z
    
#--------------------------------------------------  VQEncoder ------------------------------------------------------- 

class BinaryVQEncoder(nn.Module):
    def __init__(self, dims=[], net='ToyNet', codebook_dim=1):
        """
        multi codebook VQ encoder, each codebook contain 2 codeword (corresponding to 0/1)
        
        Args:
            dims (list): [input_dim, output_dim]
            net (str): backbone 
            num_codebooks (int)
            codebook_dim (int)
                - 1
                - output dim = num_codebooks * codebook_dim
        """
        super(BinaryVQEncoder, self).__init__()
        self.input_dim, self.output_dim = dims
        self.codebook_dim = codebook_dim
        self.num_codebooks = self.output_dim

        self.net = getattr(nets, net)(dims)

        self.embedding = nn.Parameter(
            torch.randn(self.num_codebooks, 2, self.codebook_dim)
        )
        self.embedding.data.uniform_(-1 / 2, 1 / 2)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            quantized: [B, L * d]
            embedding: [L, 2, d] 
            z_e: [B, L * d]
        """

        z_e = self.net(x)  # [B, L * d]

 
        z_e_reshaped = z_e.view(-1, self.num_codebooks, self.codebook_dim)  # [B, L, d]

        # embedding: [L, 2, d] -> [1, L, 2, d]
        # z_e_reshaped: [B, L, d] -> [B, L, 1, d]
        distances = torch.sum(
            (z_e_reshaped.unsqueeze(2) - self.embedding.unsqueeze(0)) ** 2,
            dim=-1
        )  # [B, L, 2]

        indices = torch.argmin(distances, dim=-1)  # [B, L]
        #advanced indexing: [L, 2, d] -> gather along dim=1
        # indices: [B, L] -> expand to [B, L, d]
        quantized_vectors = torch.gather(
            self.embedding.expand(z_e.size(0), -1, -1, -1),  # [B, L, 2, d]
            dim=2,
            index=indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.codebook_dim)
        ).squeeze(2)  # [B, L, d]

        # [B, L * d]
        quantized = quantized_vectors.view(-1, self.num_codebooks * self.codebook_dim)

        #Straight-Through Estimator (STE)
        quantized = z_e + (quantized - z_e).detach()

        return indices, self.embedding, quantized, z_e

    def reconstruct_from_bits(self, bits):
        """
        
        Args:
            bits: LongTensor, shape [B, L], values in {0, 1}
        
        Returns:
            reconstructed: FloatTensor, shape [B, L * D]
        """
        B, L = bits.shape
        assert L == self.num_codebooks, f"bits has {L} codebooks, but encoder expects {self.num_codebooks}"

        # [B, L, 2, D]
        expanded_embedding = self.embedding.expand(B, -1, -1, -1)

        # [B, L] -> [B, L, 1, 1]
        indices = bits.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.codebook_dim)

        # gather from dim=2 (the 2 choices: 0/1)
        quantized_vectors = torch.gather(expanded_embedding, dim=2, index=indices).squeeze(2)  # [B, L, D]

        reconstructed = quantized_vectors.view(B, self.num_codebooks * self.codebook_dim)
        return reconstructed

