def get_country_mentioned_in_context_for_fact_prompt(example: dict):
    return example['countries_in_prompt'][-1]

def get_counterfact_country(example: dict):
    return example['expected_output']

def get_fact_output(example: dict):
    return example['fact_output']