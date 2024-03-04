def return_proba_yes(values,list_words):
    """
    return the probability of the token "Yes" if it's in the list of words,
    or return the one minus the probability of the token "No" if there si only No in the list of words
    list_words: 1d list of words
    values: 1d tensor of probabilities tensors
    """
    flag_no = False
    if "Yes" in list_words:
            idx = list_words.index("Yes")
            proba_yes = values[idx] 
    elif "No" in list_words:
        idx = list_words.index("No")
        flag_no = True
        proba_No = values[idx] 

    else:
        print("No yes or no token found")
        return -1
    proba_yes=values[idx] 
    if flag_no: # if the token "no" is selected, we need to invert the probability
        proba_yes = 1-proba_No

    proba_yes=values[idx] 
    return proba_yes



