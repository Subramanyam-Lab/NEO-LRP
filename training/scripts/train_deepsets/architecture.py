import torch
import torch.nn as nn

class DeepSetArchitecture(nn.Module):
    def __init__(self, N_dim, config):        
        super(DeepSetArchitecture, self).__init__()
        self.N_dim = N_dim
        self.config = config
        activation_functions = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'LeakyReLU': nn.LeakyReLU}
 
        layers_phi = []
        layers_phi.append(nn.Linear(N_dim, config['neurons_per_layer_phi']))
        layers_phi.append(self.get_activation_function(config['activation_function']))
        for _ in range(config['num_layers_phi'] - 1):
            layers_phi.append(nn.Linear(config['neurons_per_layer_phi'], config['neurons_per_layer_phi']))
            layers_phi.append(self.get_activation_function(config['activation_function']))      
        layers_phi.append(nn.Linear(config['neurons_per_layer_phi'], config['latent_space_dimension']))
        self.phi = nn.Sequential(*layers_phi)

        layers_rho = []
        layers_rho.append(nn.Linear(config['latent_space_dimension'], config['neurons_per_layer_rho']))
        layers_rho.append(self.get_activation_function(config['activation_function']))
        for _ in range(config['num_layers_rho'] - 1):
            layers_rho.append(nn.Linear(config['neurons_per_layer_rho'], config['neurons_per_layer_rho']))
            layers_rho.append(self.get_activation_function(config['activation_function']))
        layers_rho.append(nn.Linear(config['neurons_per_layer_rho'], 1))
        layers_rho.append(self.get_activation_function(config['activation_function']))
        self.rho = nn.Sequential(*layers_rho)
        
        self.reset_parameters()

    def get_activation_function(self, name):
        activation_functions = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'LeakyReLU': nn.LeakyReLU
        }
        return activation_functions[name]()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, inputs: torch.Tensor):
        batch_size, num_customers, num_features = inputs.size()
        
        # create a mask indicating valid entries (batch_size, num_customers)
        mask = (inputs[:, :, 0] != -1.0e+04).float()
        
        # flatten inputs to apply phi to all customer features at once
        inputs_flat = inputs.view(-1, num_features)
        
        # apply phi to all inputs
        x = self.phi(inputs_flat)
        
        # reshape phi outputs back to batch structure
        x = x.view(batch_size, num_customers, -1)
        
        # apply mask to zero out phi outputs for padded entries
        x = x * mask.unsqueeze(-1)
        
        # sum over customers
        summed_tensor = x.sum(dim=1)
        
        # pass through rho
        x = self.rho(summed_tensor)
        cost = x.squeeze(-1)
        return cost


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Phi=' + str(self.phi) \
            + '\n Rho=' + str(self.rho) + ')'