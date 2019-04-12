def my_finditer(lookfor, text):
    lgth = len(lookfor)
    start_from = 0
    while True:
        begin = text.find(lookfor, start_from)
        if begin == -1:
            break
        start_from = begin+lgth
        yield (begin, begin+lgth)

def wsearch(alist, regex):
    regex = regex.lower()
    lgths = [len(s) for s in alist]
    str_ = ''.join(alist).lower()
    # spans = [item.span() for item in re.finditer(regex, str_)] # 废止
    spans = [item for item in my_finditer(regex, str_)]
    max_span = max([tup[1] for tup in spans])
    res = []; begin = 0
    for lgth in lgths:
        end = begin + lgth
        tup = (begin, end)
        try:
            isin = in_domain(tup, spans)
        except Exception as e:
            print(alist)
            print(regex)
        res.append(isin)
        begin = end
        if begin > max_span:
            break
    return res

def in_domain(tup, spans):
    res = []; result = False
    for span in spans:
        left_in = tup[0] >= span[0] and tup[0] < span[1]
        right_in = tup[1] <= span[1] and tup[1] > span[0]
        if (tup[0] <= span[0] and tup[1] > span[1]) or (tup[0] < span[0] and tup[1] >= span[1]):
            tmp = False #如果span真包含于tup, 返回false但不报错
        elif left_in^right_in: #异或。如果词跨过边界则报错，说明分词有问题
            # raise Exception('segment error detected!')
            tmp = False
        else:
            tmp = left_in and right_in
        res.append(tmp)
    result = any(res)
    return result

def analysis_one_sample(dct):
    # one line for one predicate
    words = [ins['word'] for ins in dct['postag']]
    relations = []
    object_type = []
    subject_type = []
    object_locates = []
    subject_locates = []
    object_pos = []
    subject_pos = []
    for spo in dct['spo_list']:
        relations.append(spo['predicate'])
        object_type.append(spo['object_type'])
        subject_type.append(spo['subject_type'])

        obj_locate = wsearch(words, spo['object'])
        obj_pos = []
        obj_loc = []
        for i, bool_ in enumerate(obj_locate):
            if bool_:
                obj_pos.append(dct['postag'][i]['pos'])
                obj_loc.append(i)
        object_locates.append(obj_loc)
        object_pos.append(obj_pos)

        sbj_locate = wsearch(words, spo['subject'])
        obj_pos = []
        obj_loc = []
        for i, bool_ in enumerate(sbj_locate):
            if bool_:
                obj_pos.append(dct['postag'][i]['pos'])
                obj_loc.append(i)
        subject_locates.append(obj_loc)
        subject_pos.append(obj_pos)
        
    # relations 等都是二维的表格
    res = {'relations':relations,
           'object_type':object_type,
           'subject_type':subject_type,
           'object_locates':object_locates,
           'subject_locates':subject_locates,
           'object_pos':object_pos,
           'subject_pos':subject_pos}
    return res



