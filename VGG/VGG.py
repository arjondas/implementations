import torch
import torch.nn as nn

architectures = {
  'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):
  def __init__(self, in_channels, num_classes, network_type='VGG16'):
    assert (architectures.get(network_type) is not None), 'Unknown network type. Valid options: VGG11, VGG13, VGG16, VGG19'
    '''
    network_type:
      Default: 'VGG16'
      Options:
        'VGG11'
        'VGG13'
        'VGG16'
        'VGG19'
    in_channels: number of input channels
    input format: (batch_size, channels, height, width)
    '''
    super(VGG_net, self).__init__()
    self.in_channels = in_channels
    self.conv_layers = self.create_conv_layers(architectures[network_type])

    self.fcs = nn.Sequential(
      nn.Linear(512*7*7, 4096),     ## input has to be 224, then final maxpool output will be (7x7x512) => flattened and feed to fc
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096, num_classes),
      nn.Softmax(dim=-1)
    )

  def forward(self, x):
    _, c, h, w = x.shape
    assert (w is 224), 'Input width is not 224'
    assert (h is 224), 'Input height is not 224'
    assert (c is self.in_channels), f'Given channels {c}, expected number of channels {self.in_channels}'

    x = self.conv_layers(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fcs(x)
    return x

  def create_conv_layers(self, architecture):
    layers = []
    in_channels = self.in_channels
    for x in architecture:
      if type(x) == int:        
        layers += [
          nn.Conv2d(
            in_channels=in_channels,
            out_channels=x,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
         ),
         nn.ReLU()
        ]
        in_channels = x
      
      elif x == 'M':
        layers += [
          nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2)
          )
        ]
    
    return nn.Sequential(*layers)


if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = VGG_net(3, 5, 'VGG16').to(device)
  x = torch.randn(1, 3, 224, 224)
  print(model(x).shape)