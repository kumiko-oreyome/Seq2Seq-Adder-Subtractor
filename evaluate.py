import torch,data
import utils
device = utils.get_device()
def evaluate(seq2seq,test_generator,vocab):
    #cnt = 0
    acc_cnt = 0
    total_examples = 0
    for (num1s,ops,num2s,l1s,l2s,mask1s,mask2s),(answers,ans_lens,ans_masks) in test_generator.get_batches():
        batch_size = len(num1s)
        context_vector =  seq2seq.encode(num1s,ops,num2s,l1s,l2s,mask1s,mask2s)
        decoder_input = torch.tensor([data.BOS_ID]*batch_size).view(-1,1).to(device)
        current_input = seq2seq.embeddings(decoder_input).view(-1,1,seq2seq.embedding_dim)
        current_hidden = context_vector.unsqueeze(1).transpose(0,1)
        

        max_decode_size = answers.size(1)
        preidct_each_t = []
        for t in range(max_decode_size):
            ot,ht,probs,predicts = seq2seq.predict_step(current_input ,current_hidden)
            preidct_each_t.append(predicts)
            current_input =  seq2seq.embeddings(predicts)
            current_hidden = ht
        preidct_each_t = torch.cat(preidct_each_t,1)


        import re
        predict_numbers = seq2Number(preidct_each_t, vocab,True)
        answer_numbers = seq2Number(answers, vocab,False)
        answer_numbers = [re.findall('[\-0-9]+',n)[0] for n in answer_numbers] 


        for a,b in zip(predict_numbers,answer_numbers):
            if a == b:
                acc_cnt+=1
            total_examples+=1

    print('accuracy:(%d/%d):%.3f'%(acc_cnt,total_examples,acc_cnt/total_examples))

    
# num tensors  N*MAX_SEQ_LEN
def seq2Number(num_tensors,vocab,EOS):
    assert len(num_tensors.size()) == 2
    max_seq_len = num_tensors.size(1)
    list_numbers = num_tensors.numpy().tolist()

    numbers = []
    for l in list_numbers:
        number_str = vocab.decode_sequence(l)
        numbers.append(number_str)

    number_strings = []
    for s in numbers:
        if EOS:
            end_idx = s.find(data.EOS)
            if end_idx == -1:
                end_idx = max_seq_len
        else:
            end_idx = s.rfind(data.PAD)
            if end_idx == -1:
                end_idx = max_seq_len

        number_str = s[:end_idx]
        number_strings .append(number_str)

    return number_strings

