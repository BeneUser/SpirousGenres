import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, num_blocks, num_conv_layers_per_block, kernel_size, num_first_layer_kernels, conv_stride, pool_stride,  dense_size, do_batch_norm, n_classes, config, channels=None, kernel_sizes=None, conv_strides=None, pool_strides=None):
        super().__init__()
        self.config = config #Needed for at train time initialization of dense layer input size. (Must move model to device again)

        #By default, double num of kernels each layer.
        if channels is None:
            channels = []
            currnum = num_first_layer_kernels
            for _ in range(num_blocks):
                channels.append(currnum)
                currnum *=2
        
        #By default, just keep the num of kernels/conv_strides/pool_strides the same
        if kernel_sizes is None:
            kernel_sizes = [kernel_size]*num_blocks 
        if conv_strides is None:
            conv_strides = [conv_stride]*num_blocks
        if pool_strides is None:
            pool_strides = [pool_stride]*num_blocks

        #Stack ConvBlocks together
        layers = []
        in_channels = 1
        #conv_out_size = input_size
        for i in range(num_blocks):
            layers.append(ConvBlock(num_conv_layers_per_block, in_channels, channels[i], kernel_sizes[i], conv_strides[i], pool_strides[i], do_batch_norm))
            in_channels = channels[i]
        self.conv = nn.Sequential(*layers)


        #Dense Layers at the end.
        self.dense_size = dense_size
        self.fc1 = None #Allocate at train time to get dimensions right
        self.fc2 = nn.Linear(dense_size, n_classes)


    def forward(self, x):
        #Conv Layers
        x = self.conv(x)   
        #Reshape input
        x = x.squeeze(-1)
        x = x.view(x.size(0), -1) #(Batch size, rest)
        #Allocate fc1 at train time, to get input dimensions right
        if(self.fc1 == None):
            self.fc1 = nn.Linear(in_features=x.size(1), out_features=self.dense_size)
            self.to(self.config.device)
            print("INPUT SIZE OF HIDDEN DENSE LAYER", x.size(1))
        #Dense Layers         
        x = self.fc1(x)
        x = self.fc2(x)        
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_conv_layers, in_channels, out_channels, kernel_size, conv_stride, pool_stride, do_batch_norm):
        super().__init__()
        pad = kernel_size // 2

        #Conv Layers with same structure stacked
        #Conv -> Conv -> ... -> Conv
        #Usually 1 or 2 layers.
        layers = []
        #First conv layer establishes number of channels in stack.
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=conv_stride, padding=pad))
        for i in range(num_conv_layers-1):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=conv_stride, padding=pad))
        
        self.conv = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.do_batch_norm = do_batch_norm #En/disable batch norm

    def forward(self, x):
        x = self.conv(x)
        if(self.do_batch_norm):
            x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x
