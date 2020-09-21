






import torch
import torch.nn as nn
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText






# use cuda
def gpu(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(device), device

# register kwargs for some functions
def register_kwargs(**kwargs):
    def decorator(func):
        def wrapper(*args, **nkwargs):
            nkwargs.update(kwargs)
            results = func(*args, **nkwargs)
            return results
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
