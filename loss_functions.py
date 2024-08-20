from activation_functions import softmax

def cross_entropy_loss(logits, labels):
  # predictions.shape => [batch_size, 1, 10]
  # labels.shape => [batch_size]
  soft_preds = softmax(logits)
  log_preds = soft_preds.log()
  loss = -log_preds[range(logits.shape[0]), labels].mean()
  
  # manual backprop
  dlogits = soft_preds.clone()
  dlogits[range(logits.shape[0]), labels] -= 1
  dlogits /= logits.shape[0]
  
  return loss, dlogits