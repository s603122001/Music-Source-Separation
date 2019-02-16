import torch
import torch.nn as nn

class ARC(nn.Module):
    def __init__(self, sources = 2):
        super(ARC, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(1025, 1024, kernel_size = 3, padding = 1), name = "weight")
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(1024, 512, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.conv3 = nn.utils.weight_norm(nn.Conv1d(512, 256, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.Tconv1 = nn.utils.weight_norm(nn.ConvTranspose1d(256, 512, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv2 = nn.utils.weight_norm(nn.ConvTranspose1d(512, 1024, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv3 = nn.utils.weight_norm(nn.Conv1d(1024, 1025 * sources, kernel_size = 3, stride = 1, padding = 1), name = "weight")
        self.gru1 = nn.GRU(1024, hidden_size = 1024, num_layers = 1, batch_first = True)
        self.gru2 = nn.GRU(512, hidden_size = 512, num_layers = 1, batch_first = True)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x):
        # TODO: check skip connection before or after activation
        # TODO: check operational direction of GRU

        x_s1 = self.leaky_relu(self.conv1(x))
        x_s2 = self.leaky_relu(self.conv2(x_s1))
        x = self.leaky_relu(self.conv3(x_s2))
        
        #GRU layer output shape (seq_len, batch, num_directions * hidden_size)
        x_s1 = x_s1.permute(0, 2, 1)
        x_s1 = self.leaky_relu(self.gru1(x_s1)[0])
        x_s1 = x_s1.permute(0, 2, 1)
        
        x_s2 = x_s2.permute(0, 2, 1)
        x_s2 = self.leaky_relu(self.gru2(x_s2)[0])
        x_s2 = x_s2.permute(0, 2, 1)
        
        x = self.leaky_relu(self.Tconv1(x))
        x = torch.add(x, x_s2)
        x = self.leaky_relu(self.Tconv2(x))
        x = torch.add(x, x_s1)
        x = self.leaky_relu(self.Tconv3(x))
        return x
    
class Enhancement(nn.Module):
    def __init__(self):
        super(Enhancement, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(1025, 1024, kernel_size = 3, padding = 1), name = "weight")
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(1024, 512, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.conv3 = nn.utils.weight_norm(nn.Conv1d(512, 256, kernel_size = 3, stride = 2, padding = 1), name = "weight")
        self.Tconv1 = nn.utils.weight_norm(nn.ConvTranspose1d(256, 512, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv2 = nn.utils.weight_norm(nn.ConvTranspose1d(512, 1024, kernel_size = 3, stride = 2, padding = 1, output_padding = 1), name = "weight")
        self.Tconv3 = nn.utils.weight_norm(nn.Conv1d(1024, 1025, kernel_size = 3, stride = 1, padding = 1), name = "weight")
        self.skip_conv1 = nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size = 3, padding = 1), name = "weight")
        self.skip_conv2 = nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size = 3, padding = 1), name = "weight")
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x):
        # TODO: check skip connection before or after activation

        x_s1 = self.leaky_relu(self.conv1(x))
        x_s2 = self.leaky_relu(self.conv2(x_s1))
        x = self.leaky_relu(self.conv3(x_s2))
        x_s1 = self.leaky_relu(self.skip_conv1(x_s1))
        x_s2 = self.leaky_relu(self.skip_conv2(x_s2))
        
        x = self.leaky_relu(self.Tconv1(x))
        x = torch.add(x, x_s2)
        x = self.leaky_relu(self.Tconv2(x))
        x = torch.add(x, x_s1)
        x = self.leaky_relu(self.Tconv3(x))
        return x