import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

###############################################################################

def kaiming_init(m):
    
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

#-----------------------------------------------------------------------------#
            
def normal_init(m):
    
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
###############################################################################

class RelevanceVector(nn.Module):
    
    def __init__(self, z_dim):
        
        super(RelevanceVector, self).__init__()
        
        self.rvlogit = nn.Parameter(0.001*torch.randn(z_dim))
        #self.rvlogit = nn.Parameter(5.0*torch.ones(z_dim))

    def forward(self):
        
        rv = torch.sigmoid(self.rvlogit)
        
        return self.rvlogit, rv

###############################################################################
        
class Discriminator(nn.Module):
    
    '''
    returns (n x 2): Let D1 = 1st column, D2 = 2nd column, then the meaning is
      D(z) (\in [0,1]) = exp(D1) / ( exp(D1) + exp(D2) )
      
      so, it follows: log( D(z) / (1-D(z)) ) = D1 - D2
    '''
    
    ####
    def __init__(self, z_dim):
        
        super(Discriminator, self).__init__()
        
        self.z_dim = z_dim
        
        self.net = nn.Sequential(
          nn.Linear(z_dim, 1000), nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 1000),  nn.LeakyReLU(0.2, True),
          nn.Linear(1000, 2),
        )
        
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)


    ####
    def forward(self, z):
        
        return self.net(z)
    

#-----------------------------------------------------------------------------#
        
class Encoder1(nn.Module):
    
    '''
    encoder architecture for the "dsprites" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Encoder1, self).__init__()
        
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc5 = nn.Linear(64*4*4, 128)
        self.fc6 = nn.Linear(128, 2*z_dim) 
        
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc5(out))
        stats = self.fc6(out)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        std = torch.sqrt(torch.exp(logvar))
        
        return mu, std, logvar


#-----------------------------------------------------------------------------#
        
class Decoder1(nn.Module):
    
    '''
    decoder architecture for the "dsprites" data
    '''
    
    ####
    def __init__(self, z_dim=10):
        
        super(Decoder1, self).__init__()
        
        self.z_dim = z_dim
        
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 4*4*64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
       
        # initialize parameters
        self.weight_init()


    ####
    def weight_init(self, mode='normal'):
        
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])


    ####
    def forward(self, z):
        
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv5(out))
        x_recon = self.deconv6(out)
            
        return x_recon


###############################################################################

#class FactorVAE2(nn.Module):
#    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
#    def __init__(self, z_dim=10):
#        super(FactorVAE2, self).__init__()
#        self.z_dim = z_dim
#        self.encode = nn.Sequential(
#            nn.Conv2d(3, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(32, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(32, 64, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(64, 64, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(64, 256, 4, 1),
#            nn.ReLU(True),
#            nn.Conv2d(256, 2*z_dim, 1)
#        )
#        self.decode = nn.Sequential(
#            nn.Conv2d(z_dim, 256, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(256, 64, 4),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(64, 64, 4, 2, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(64, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(32, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(32, 3, 4, 2, 1),
#        )
#        self.weight_init()
#
#    def weight_init(self, mode='normal'):
#        if mode == 'kaiming':
#            initializer = kaiming_init
#        elif mode == 'normal':
#            initializer = normal_init
#
#        for block in self._modules:
#            for m in self._modules[block]:
#                initializer(m)
#
#    def reparametrize(self, mu, logvar):
#        std = logvar.mul(0.5).exp_()
#        eps = std.data.new(std.size()).normal_()
#        return eps.mul(std).add_(mu)
#
#    def forward(self, x, no_dec=False):
#        stats = self.encode(x)
#        mu = stats[:, :self.z_dim]
#        logvar = stats[:, self.z_dim:]
#        z = self.reparametrize(mu, logvar)
#
#        if no_dec:
#            return z.squeeze()
#        else:
#            x_recon = self.decode(z)
#            return x_recon, mu, logvar, z.squeeze()


###############################################################################

#class FactorVAE3(nn.Module):
#    """Encoder and Decoder architecture for 3D Faces data."""
#    def __init__(self, z_dim=10):
#        super(FactorVAE3, self).__init__()
#        self.z_dim = z_dim
#        self.encode = nn.Sequential(
#            nn.Conv2d(1, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(32, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(32, 64, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(64, 64, 4, 2, 1),
#            nn.ReLU(True),
#            nn.Conv2d(64, 256, 4, 1),
#            nn.ReLU(True),
#            nn.Conv2d(256, 2*z_dim, 1)
#        )
#        self.decode = nn.Sequential(
#            nn.Conv2d(z_dim, 256, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(256, 64, 4),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(64, 64, 4, 2, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(64, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(32, 32, 4, 2, 1),
#            nn.ReLU(True),
#            nn.ConvTranspose2d(32, 1, 4, 2, 1),
#        )
#        self.weight_init()
#
#    def weight_init(self, mode='normal'):
#        if mode == 'kaiming':
#            initializer = kaiming_init
#        elif mode == 'normal':
#            initializer = normal_init
#
#        for block in self._modules:
#            for m in self._modules[block]:
#                initializer(m)
#
#    def reparametrize(self, mu, logvar):
#        std = logvar.mul(0.5).exp_()
#        eps = std.data.new(std.size()).normal_()
#        return eps.mul(std).add_(mu)
#
#    def forward(self, x, no_dec=False):
#        stats = self.encode(x)
#        mu = stats[:, :self.z_dim]
#        logvar = stats[:, self.z_dim:]
#        z = self.reparametrize(mu, logvar)
#
#        if no_dec:
#            return z.squeeze()
#        else:
#            x_recon = self.decode(z)
#            return x_recon, mu, logvar, z.squeeze()



