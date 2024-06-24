def logits_to_logit_diff(model, logits, first_token, second_token):
    first_token_index = model.to_single_token(first_token)
    second_token_index = model.to_single_token(second_token)
    return logits[0, -1, first_token_index] - logits[0, -1, second_token_index]

def token_logit(model, logits, token):
    token_index = model.to_tokens(token)[0][1].item()
    return logits[0, -1, token_index]