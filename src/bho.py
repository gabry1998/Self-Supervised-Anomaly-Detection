from self_supervised.models import PeraNet

peranet = PeraNet(
    intermediate_outputs=['layer2','layer3'],
    latent_space_layers=3,
    latent_space_layers_base_dim=512,
    num_classes=3,
    memory_bank_dim=50
)
peranet.compile(
    learning_rate=0.03,
    epochs=30
)