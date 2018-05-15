import numpy as np
def get_device():
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
import data
# raw data -> preprocess data <--> inverse
# prerocess data -->  model datas (X,y)  <--> inverse 
# model datas --> batch datas  

             
#TODO REVERSE

                                            # 1998~ -999 + EOS = 5 tokens
def generate_examples(num_examples,MAX_DIGIT=3,MAX_ANS_LEN=5):
    questions = []
    expected = []
    seen = set()

    while len(questions) < num_examples:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, MAX_DIGIT + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        op = np.random.choice(list('+-'))
        q = '{}{}{}'.format(a,op,b)
        query = q 
        if op == '+':
            ans = str(a + b)
        else:
            ans = str(a - b)
        #if REVERSE:
        #    q_len =   len(query.strip())
        #    ans_len = len(ans.strip())
        #    query = query[q_len-1::-1]+query[q_len::]
        #    ans =  ans[ans_len-1::-1]+ ans[ans_len::]
        questions.append(query)
        expected.append(ans)
    return questions,expected

def get_numbers_and_op(expression):
    op = '+'
    if op  not in expression  :
        op = '-'
    op_index = expression.find(op)
    num1 = expression[0:op_index]
    num2 = expression[op_index+1:]
    return num1,op,num2

def padding_sequence(x,vocab,max_len=3):
    assert len(x)<=max_len
    return x+[vocab.get_cid(data.PAD)]*(max_len-len(x))



# turn 
#def preprocessing(expression,answers,MAX_DIGIT,MAX_ANS_LEN=5,REVERSE):
#    pass
#


#import functools
#_mask_func = lambda max_len,real_len:[1 if i<real_len else 0 for i in range(max_len)]
#mask_func = functools.partial(_mask_func, MAX_DIGIT)
def get_mask(seq_len,max_len):
    return [1 if i<seq_len else 0 for i in range(max_len)]


def expression2Xs(expressions,vocab,MAX_DIGIT=3):
    Xs = []
    lens = []
    masks = []
    for q in expressions:
        n1,op,n2=get_numbers_and_op(q)
        l1,l2=list(map(len,[n1,n2]))
        mask1,mask2 = get_mask(l1,MAX_DIGIT),get_mask(l2,MAX_DIGIT)
        seq1,seq2 = vocab.encode_string(n1),vocab.encode_string(n2)
        Xs.append((padding_sequence(seq1,vocab,MAX_DIGIT),[vocab.get_cid(op)],padding_sequence(seq2,vocab,MAX_DIGIT)))
        lens.append((l1,l2))
        masks.append((mask1,mask2))
    return Xs,lens,masks

def answer2Ys(answers,vocab,MAX_DIGIT=5):
    ys = []
    lens = []
    masks = []
    for ans in answers:
        eos_answers = ans+data.EOS
        l= len(eos_answers)
        mask =get_mask(l,MAX_DIGIT)
        seq = vocab.encode_string(eos_answers)
        ys.append(padding_sequence(seq,vocab,MAX_DIGIT))
        lens.append(l)
        masks.append(mask)
    return ys,lens,masks

