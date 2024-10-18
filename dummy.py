def validate_encoding(inputs, encoding):
    '''
    tokenized_encoding = mmm.tokenization(input_prompt, **mmm.processor_kwargs)
    if validate_encoding(inputs, tokenized_encoding):
        raise Exception('inputs != tokenized_encoding')
    '''
    input_ids = inputs['input_ids'].detach().cpu().numpy()
    input_ids_shape = input_ids.shape

    enc_ids = encoding.ids
    num_ids = len(enc_ids)

    if input_ids_shape[-1] != num_ids:
        return True
    
    for (crit, comp) in zip(input_ids[0], enc_ids):
        if crit != comp:
            return True
        
    return False