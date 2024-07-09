from typing import List
import tqdm
import numpy
import torch


def generate_mean_attn_activation_values(model, keys: List[str], abc_examples: List[dict], prompt_key: str, template_words: List[dict]):
  torch.set_grad_enabled(False)
  mean_activation_values = {}
  
  for abc_example in tqdm.tqdm(abc_examples):
    abc_tokens = model.to_tokens(abc_example[prompt_key])

    _, abc_cache = model.run_with_cache(abc_tokens)

    word_positions = get_word_pos_to_token_pos_mapping(model, template_words, abc_example, prompt_key)

    for word_position in word_positions:
      if word_position['word_pos'] not in mean_activation_values:
        mean_activation_values[word_position['word_pos']] = {}

      for key in keys:
        if key not in mean_activation_values[word_position['word_pos']]:
          mean_activation_values[word_position['word_pos']][key] = numpy.zeros_like(abc_cache[key][-1, word_position['token_pos'], :, :])
        mean_activation_values[word_position['word_pos']][key] += abc_cache[key][-1, word_position['token_pos'], :, :].cpu().detach().numpy()

  for word_pos in mean_activation_values.keys():
    for key in keys:
      mean_activation_values[word_pos][key] /= len(abc_examples)
  
  return mean_activation_values

def get_word_pos_to_token_pos_mapping(model, template_words: List[dict], abc_example: dict, prompt_key: str):
  word_positions = []
  for template_word in template_words:
    if template_word['word'] == 'city name':
      token_positions = find_token_appearance(model, abc_example[prompt_key], ' ' + abc_example['cities_in_prompt'][0])
      token_positions.extend(find_token_appearance(model, abc_example[prompt_key], ' ' + abc_example['cities_in_prompt'][1]))
      assert len(token_positions) == len(template_word['position']), "number of city name occurencies in text doesn't match template"

      for word_position, token_position in zip(template_word['position'], token_positions):
        word_positions.append({ 'word_pos': word_position, 'token_pos': token_position })

    elif template_word['word'] == 'country name':
      assert len(abc_example['countries_in_prompt']) == 1, "number of countries is too big"
      token_positions = find_token_appearance(model, abc_example[prompt_key], ' ' + abc_example['countries_in_prompt'][0])
      assert len(token_positions) == 1, "number of country occurences is too big"

      word_positions.append({ 'word_pos': template_word['position'][0], 'token_pos': token_positions[0] })

    else:
      token_positions = find_token_appearance(model, abc_example[prompt_key], template_word['word'] )
      assert len(token_positions) == len(template_word['position'])\
        , f"number of occurencies in text doesn't match template. prompt: {abc_example[prompt_key]}, token_positions: {token_positions}, template_word: {template_word}"

      for word_position, token_position in zip(template_word['position'], token_positions):
        word_positions.append({ 'word_pos': word_position, 'token_pos': token_position })
  
  return word_positions

def find_token_appearance(model, prompt, token):
  token_embeddings = model.to_tokens(token)[0][1:].detach().tolist()
  prompt_embeddings = model.to_tokens(prompt)[0].detach().tolist()

  positions = []

  for i in range(len(prompt_embeddings) - len(token_embeddings) + 1):
    if prompt_embeddings[i:i+len(token_embeddings)] == token_embeddings:
      positions.append(i+len(token_embeddings)-1)

  return positions