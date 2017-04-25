import string


def normalize(desc):
    nopunct = [w for w in desc if w not in string.punctuation]
    return ''.join(nopunct).lower()
