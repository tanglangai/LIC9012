import numpy as np


def word_level_eval(preds, raw_list, show_worst = True):
    assert len(preds) == len(raw_list)
    sub_recall_list = []
    ob_recall_list = []
    n_sub_list = []
    n_ob_list = []
    for pred, raw_inst in zip(preds, raw_list):
        sub_words = extract_word(raw_inst['raw_text'], pred[:, 0])
        ob_words = extract_word(raw_inst['raw_text'], pred[:, 1])
        real_subs = set([spo['subject'] for spo in raw_inst['spo_list']])
        real_obs = set([spo['object'] for spo in raw_inst['spo_list']])
        sub_correct = real_subs.intersection(sub_words)
        ob_correct = real_obs.intersection(ob_words)
        sub_recall = len(sub_correct)/len(real_subs)
        ob_recall = len(ob_correct)/len(real_obs)
        sub_recall_list.append(sub_recall)
        ob_recall_list.append(ob_recall)
        n_sub_list.append(len(sub_words))
        n_ob_list.append(len(ob_words))
    mean_sub_recall = np.mean(sub_recall_list)
    mean_ob_recall = np.mean(ob_recall_list)
    mean_nsub = np.mean(n_sub_list)
    mean_nob = np.mean(n_ob_list)
    print('='*80)
    print('mean_sub_recall: ', mean_sub_recall)
    print('mean ob recall: ', mean_ob_recall)
    print('mean sub number: ', mean_nsub)
    print('mean ob number: ', mean_nob)

    if show_worst:
        worst_first = np.argsort(sub_recall_list)
        preds = [preds[i] for i in worst_first[:15]]
        raw_insts = [raw_list[i] for i in worst_first[:15]]
        ii = 0
        for pred, raw_inst in zip(preds, raw_insts):
            sub_words = extract_word(raw_inst['raw_text'], pred[:, 0])
            ob_words = extract_word(raw_inst['raw_text'], pred[:, 1])
            real_subs = set([spo['subject'] for spo in raw_inst['spo_list']])
            real_obs = set([spo['object'] for spo in raw_inst['spo_list']])
            print(raw_inst['raw_text'])
            print('Real subjects: '+' '.join(real_subs))
            print('Pred subjects: '+' '.join(sub_words))
            print('Real objects: ' + ' '.join(real_obs))
            print('Pred objects: ' + ' '.join(ob_words))
            ii += 1
            if ii > 10:
                break

def extract_word(text, bools, return_location = False):
    res = []
    memory = []
    inword = False
    pointer = 0
    locations = {}
    for char, bool in zip(text, bools[:len(text)]):
        if bool:
            memory.append(char)
            if not inword:
                starter = pointer
            inword = True
        else:
            if inword:
                word = ''.join(memory)
                res.append(word)
                locations[word] = starter
                memory = []
                inword = False
        pointer += 1
    if inword:
        word = ''.join(memory)
        res.append(word)
    if return_location:
        return set(res), locations
    else:
        return set(res)
