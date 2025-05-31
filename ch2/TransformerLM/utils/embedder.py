from torch import nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Initialize the Embedder module.
        
        Args:
            vocab_size (int): The size of the vocabulary (one-hot encoding number of words).
            d_model (int): The dimensionality of the model.
        """
        super().__init__()
        # Initialize the embedding layer with the given vocabulary size and model dimension.
        # The embedding layer maps each word in the vocabulary to a dense vector of size d_model.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)