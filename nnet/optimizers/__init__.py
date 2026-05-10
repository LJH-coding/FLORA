# Optimizers
from .adam import Adam
from .sam import SAM

# Optimizers Dictionary
optim_dict = {
    "Adam": Adam,
    "SAM": SAM,
}