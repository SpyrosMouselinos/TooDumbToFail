from torch import nn



class StandardEncoder(nn.Linear):
    def __init__(self, d_input: int, d_model: int, bias: bool = True) -> None:
        super().__init__(in_features=d_input, out_features=d_model, bias=bias)
        self.d_input = d_input
        self.d_model = d_model





