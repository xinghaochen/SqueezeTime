import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import TypeVar, Union, Tuple
from mmaction.registry import MODELS

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

def get_inplanes():
    
    return [64, 128, 256, 512]
    
class SpatialConv(nn.Module):
    """
    Inter-temporal Object Interaction Module (IOI)
    """
    def __init__(self, dim_in, dim_out, pos_dim=7):
        super(SpatialConv, self).__init__()
        
        self.short_conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, groups=1)
        self.glo_conv = nn.Sequential(nn.Conv2d(dim_in, 16, kernel_size=3, stride=1, padding=1, groups=1), 
                                    nn.BatchNorm2d(16), nn.ReLU(inplace=True), 
                                    nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3), 
                                    nn.BatchNorm2d(16), nn.ReLU(inplace=True), 
                                    nn.Conv2d(16, dim_out, kernel_size=3, stride=1, padding=1, groups=1), nn.Sigmoid())
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, pos_dim,pos_dim))
        
        nn.init.kaiming_normal_(self.pos_embed, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, param):
        
        x_short = self.short_conv(x)
        x = x*param
        
        for i in range(len(self.glo_conv)):
            
            if i==3:
                _,_,H,W=x.shape
                
                if self.pos_embed.shape[2] !=H or self.pos_embed.shape[3]!=W:
                    pos_embed = F.interpolate(self.pos_embed, size=((H,W)), scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None)
                else:
                    pos_embed = self.pos_embed
                
                x = x + pos_embed
            
            x = self.glo_conv[i](x)
        
        return x_short*x
    
class Conv2d(nn.Module):
    """
    Channel-Time Learning Module (CTL)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros', 
        pos_dim = 7):
        super(Conv2d, self).__init__()
        
        self.stride = stride
        self.param_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels,
                      in_channels,
                      1,
                      stride=1,
                      padding=1 // 2,
                      bias=False), 
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid())
        
        self.temporal_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.spatial_conv = SpatialConv(dim_in=in_channels, dim_out=out_channels, pos_dim=pos_dim)
    
    def forward(self, x):
        
        param = self.param_conv(x)
        x = self.temporal_conv(param*x) + self.spatial_conv(x, param)
        
        return x



def conv3x3x3(in_planes, out_planes, stride=1, pos_dim=7):

    return Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False,
                     pos_dim=pos_dim)

def conv1x1x1(in_planes, out_planes, stride=1):
    
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    """
    Channel-Time Learning (CTL) Block
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut_conv=None, pos_dim=7):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 =conv3x3x3(planes, planes,pos_dim=pos_dim)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut_conv = shortcut_conv
        
        self.stride = stride
        if stride != 1:
            self.downsample = nn.Sequential( nn.Conv2d(in_planes, in_planes, kernel_size=2, stride=2, groups=in_planes), nn.BatchNorm2d(in_planes))
            
    def forward(self, x):
        
        if self.stride != 1:
            x = self.downsample(x)
            
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    Channel-Time Learning (CTL) Block
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut_conv=None, pos_dim=7):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3x3(planes, planes, pos_dim=pos_dim)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut_conv = shortcut_conv
        self.stride = stride
        
        if stride !=1:
            self.downsample = nn.Sequential( nn.Conv2d(in_planes, in_planes, kernel_size=2, stride=2, groups=in_planes), nn.BatchNorm2d(in_planes))
    
    def forward(self, x):
        
        if self.stride != 1:
            x=self.downsample(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400, 
                 dropout=0.2, 
                 freeze_bn=False, 
                 spatial_stride=[1,2,2,2], 
                 pos_dim=[64,32,16,8]):
        super().__init__()
        
        self.freeze_bn = freeze_bn
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.dropout = dropout
        
        self.conv1 = nn.Conv2d(n_input_channels,
                               self.in_planes,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               groups=1,
                               bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, stride=spatial_stride[0], pos_dim=pos_dim[0])
        
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=spatial_stride[1], pos_dim=pos_dim[1])
        
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=spatial_stride[2], pos_dim=pos_dim[2])
        
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=spatial_stride[3], pos_dim=pos_dim[3])

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        
        out = F.avg_pool2d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1,pos_dim=7):
        
        shortcut = None
        if  self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride=1),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride, shortcut_conv=shortcut, pos_dim=pos_dim))
        
        self.in_planes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, pos_dim=pos_dim))

        return nn.Sequential(*layers)

    def forward(self, x): 
        
        if isinstance(x, dict):
            x = x["video"]
            
        N,C,T,H,W=x.shape
        x=x.view(N,-1,H,W)                                                                                                                                                   
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def train(self, mode=True):
        
        freeze_bn = self.freeze_bn
        freeze_bn_affine = self.freeze_bn
        super(ResNet, self).train(mode)
        
        if freeze_bn:
            print ("Freezing Mean/Var of BatchNorm2D.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        
        if freeze_bn_affine:
            print ("Freezeing Weight/Bias of BatchNorm2D.")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True

def generate_model(model_depth, **kwargs):
    
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def load_ckpts(model,load_path):
    
    pretrain = torch.load(load_path, map_location='cpu')
    adapted_weights={}
    
    if 'state_dict' in pretrain.keys():
        pretrain=pretrain['state_dict']
    elif 'model' in pretrain.keys():
        pretrain=pretrain['model']
    
    if not hasattr(model, 'module'):
        for name,module in model.named_parameters():
            
            if name in pretrain.keys() and pretrain[name].data.shape == module.data.shape:
                adapted_weights[name]=pretrain[name].data
                
        model.load_state_dict(adapted_weights,strict=False)
    else:
        for name,module in model.module.named_parameters():
            
            if name in pretrain.keys() and pretrain[name].data.shape == module.data.shape:
                adapted_weights[name]=pretrain[name].data
        
        model.module.load_state_dict(adapted_weights,strict=False)
    
    print('load backbone finetue ckpts done!')
    
    return model

@MODELS.register_module()
class SqueezeTime(nn.Module):
    """
    Build SqueezeTime Model
    """
    def __init__(self, depth=50, widen_factor=1.0, dropout=0.5, input_channels=48, n_classes=400, load=None, freeze_bn=False, spatial_stride=[1, 2, 2, 2], pos_dim=[64, 32, 16 ,8]): 
        super(SqueezeTime,self).__init__()
        
        self.net = generate_model(depth, widen_factor=widen_factor, dropout=dropout, n_input_channels=input_channels, n_classes=n_classes, freeze_bn=freeze_bn, spatial_stride=spatial_stride, pos_dim=pos_dim).cuda()
        
        if load is not None:
            self.net = load_ckpts(self.net, load)
            
    def forward(self, x):
        x = self.net(x)
        
        return x
    
if __name__ == '__main__':
    
    import time
    from ptflops import get_model_complexity_info
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    
    t = 16
    h = 224
    batchsize = 256
    
    model = generate_model(50, widen_factor=1.0, dropout=0.5, n_input_channels=int(t*3), n_classes=400, pos_dim=[56, 28, 14, 7]).cuda()
    model.eval()
    
    # Calculate FLOPs and Params 
    tensor = torch.rand(1, 3, t, h, h).cuda()
    
    flops = FlopCountAnalysis(model, tensor)
    
    print("Fvcore FLOPs: ", flops.total()/1e9)
    print("Params: ",parameter_count_table(model, 1))
    
    # Calculate GPU forward time
    model = model.cuda()
    tensor = torch.rand(1, 3, t, h, h).cuda()
    
    for i in range(10):
        model(tensor)
        
    start=time.time()
    for i in range(100):
        model(tensor)
        
    print('GPU: forward time: ', time.time()-start)

    # Calculate GPU throughput 
    tensor = torch.rand(batchsize, 3, t, h, h).cuda()
    
    for i in range(10):
        model(tensor)
        
    repetitions = 100
    total_time = 0
    
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            starter.record()
            model(tensor)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
            
    Throughput = (repetitions*batchsize)/total_time
    
    print('Throughput: ',Throughput)
    print("Max allocated memory:", torch.cuda.max_memory_allocated()/1e9)

    # Calculate CPU forward time
    model=model.cpu()
    tensor = torch.rand(1, 3, t, h, h)
    
    for i in range(10):
        model(tensor)
        
    start=time.time()
    for i in range(10):
        model(tensor)
    
    print('CPU: forward time: ',time.time()-start)
