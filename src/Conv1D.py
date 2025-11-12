import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, in_c=1, out_c=64, k=7, use_pool=True, n_classes=10):
        super().__init__()
        pad = k // 2  
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=k, stride=2, padding=pad)
        self.use_pool = use_pool
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(out_c, n_classes)

    def forward(self, x):              
        x = nn.functional.relu(self.conv(x))       
        x = self.pool(x)           
        x = self.gap(x)       
        x = x.squeeze(-1)            
        x = self.fc(x)                
        return x