import timm

def model(model_name, pretrained, num_classes):
    return timm.create_model(model_name, pretrained, num_classes=num_classes,  )

def load_model(model_name:str, path:str, num_classes:int):
    return timm.create_model(model_name, num_classes=num_classes, checkpoint_path=path )