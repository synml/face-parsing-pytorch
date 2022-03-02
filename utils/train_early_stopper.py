def train_early_stopper():
    with open('train_early_stopper.ini', 'r', encoding='utf-8') as f:
        flag = f.read().strip()

    if flag == '0':
        return False
    elif flag == '1':
        with open('train_early_stopper.ini', 'w', encoding='utf-8') as f:
            f.write('0')
        return True
    else:
        raise ValueError('Wrong flag value.')
