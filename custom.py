from tensorflow.keras import losses


# --- Custom Loss & Metric Functions
def WMAE(weight=1e-3):
    def loss(y_true, y_pred):
        return losses.MAE(y_true, y_pred) * weight
    return loss

def WMSE(weight=3e-8):
    def loss(y_true, y_pred):
        return losses.MSE(y_true, y_pred) * weight
    return loss

def BCE(from_logits=True):
    def loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred, from_logits)
    return loss