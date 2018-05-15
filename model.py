from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch,data,utils
from torch import nn


device = utils.get_device()

def dynamic_rnn(rnn_cell,padded_sequences,seq_lens,hts):
    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    padded_sequences =  padded_sequences[perm_idx]
                                                          #seq length: type is not a tensor
    packed_input = pack_padded_sequence(padded_sequences, seq_lens.data.cpu().numpy(),batch_first=True)
    packed_output, ht = rnn_cell(packed_input,hts)
    #pad_packed_sequence :只會返回seq_)en最常長度的那個padding值而已..
    output, _ = pad_packed_sequence(packed_output,batch_first=True)
    _,unsort_idx= perm_idx.sort(0)
    output,ht = output[unsort_idx],ht.squeeze(0)[unsort_idx]
    return output, ht

class Seq2seqCalc(nn.Module):
    #default multiplicative attention
    def __init__(self,embedding_dim,vocab_size,digit_rnn_units,decoder_rnn_units):
        super(Seq2seqCalc , self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.digit_rnn_units = digit_rnn_units #encoder
        self.decoder_rnn_units = decoder_rnn_units

        self.build()


    def get_hyper(self):
        return self.embedding_dim,self.vocab_size,self.digit_rnn_units,self.decoder_rnn_units


    def build(self):
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim).to(device)
        self.digit_rnn = nn.GRU(self.embedding_dim, self.digit_rnn_units,batch_first=True).to(device)
        self.init_gru_parameters(self.digit_rnn)
        #self.attention_W = torch.Tensor((self.embedding_dim,self.digit_rnn_units)).to(device)-->不行 這樣會以為是2個值(self.embedding_dim,self.digit_rnn_units)
        self.attention_W = nn.Parameter(torch.ones(()).new_empty((self.embedding_dim,self.digit_rnn_units),device=device))
        torch.nn.init.uniform_(self.attention_W,0.1,-0.1)
        
        
        #project to decoder dimension
        self.context_W = nn.Parameter(torch.ones(()).new_empty((self.digit_rnn_units,self.decoder_rnn_units),device=device))
        torch.nn.init.uniform_(self.context_W ,0.1,-0.1)


        self.decoder = nn.GRU(self.embedding_dim, self.decoder_rnn_units,batch_first=True).to(device)
        self.init_gru_parameters(self.decoder)

        self.predict_W = nn.Parameter(torch.ones(()).new_empty((self.decoder_rnn_units,self.vocab_size),device=device))
        torch.nn.init.uniform_(self.predict_W)


    def init_gru_state(self,batch_size,hidden_size):
        return torch.zeros((1,batch_size,hidden_size),requires_grad=False)



    
    def init_gru_parameters(self,gru):
        W_ir,W_iz,W_in = gru.weight_ih_l0.chunk(3, 0)# i input vector
        W_hr,W_hz,W_hn = gru.weight_hh_l0.chunk(3, 0) # h hidden vector
        #orthogonal initialization
        nn.init.orthogonal_(W_ir)
        nn.init.orthogonal_(W_iz)
        nn.init.orthogonal_(W_in)
        nn.init.orthogonal_(W_hr)
        nn.init.orthogonal_(W_hz)
        nn.init.orthogonal_(W_hn)

    #              inp : N*C
    def maskNLLLoss(self,inp, target, mask):
        nTotal = mask.sum()
        logits = torch.gather(inp, 1, target.view(-1, 1))
        assert inp.size(0) == torch.sum(logits.gt(0)).item()
        crossEntropy = -torch.log(logits)
        mask = mask.view(-1,1).byte()
        # 當mask 全部都為0的時候會返回empty tensor , empty tensor無法mean()
        masked = crossEntropy.masked_select(mask)

        if masked.numel() == 0:
            loss = torch.tensor(0.0)
        else:
            loss =  masked.mean()
        return loss, nTotal



    def encode(self,num1s,ops,num2s,l1s,l2s,mask1s,mask2s):
        #N*S*E # N*1*E
        emb1,emb_op,emb2= self.embeddings(num1s),self.embeddings(ops),self.embeddings(num2s)      

        ht1s,_ = dynamic_rnn(self.digit_rnn,emb1,l1s,self.init_gru_state(emb1.size(0),self.digit_rnn_units))
        ht2s,_ = dynamic_rnn(self.digit_rnn,emb2,l2s,self.init_gru_state(emb2.size(0),self.digit_rnn_units))

        ht1_max_seqlen =  ht1s.size(1)
        ht2_max_seqlen =  ht2s.size(1)
        
        mask1s = mask1s[:,:ht1_max_seqlen]
        mask2s = mask2s[:,:ht2_max_seqlen]

        #concat for attention
        # N*CAT*H
        h_concat = torch.cat((ht1s,ht2s),1)
        #N*CAT
        h_masks = torch.cat((mask1s,mask2s),1)
        h_masks = h_masks.unsqueeze(1)

        # N*1*H                N*1*E       E*H
  
        _att1s = torch.matmul(emb_op,self.attention_W)
        # N*1*CAT                     N*1*H,N*CAT*H
        att1s = torch.matmul(_att1s,h_concat.transpose(1,2))
        att1s = att1s*h_masks.float()
    
        att_weights = nn.Softmax(2)(att1s)
        att_weights = att_weights.transpose(1,2)
        #有的時候會有NAN的情況? 如果沒有設定初始化的值

        _context_vector = torch.sum(att_weights*h_concat,1)
        context_vector = torch.matmul(_context_vector,self.context_W)

        return context_vector

    # current_input : N*1*E , current_hidden:1*N*H
    def predict_step(self,current_input,current_hidden):
        ot,ht = self.decoder(current_input,current_hidden)
        ot = ot[:,-1,:]
        # N*V
        logits =  torch.matmul(ot,self.predict_W)
        probs =  torch.nn.Softmax(1)(logits)
        _ , predicts = torch.topk(probs,1,1)
        return ot,ht,probs,predicts

    def forward(self, Xs,ys):
       
        (num1s,ops,num2s,l1s,l2s,mask1s,mask2s),(answers,ans_lens,ans_masks) = Xs,ys

        batch_size = num1s.size(0)

        context_vector = self.encode(num1s, ops, num2s, l1s, l2s, mask1s, mask2s)

        #decode
        decoder_input = torch.tensor([data.BOS_ID]*batch_size).view(-1,1).to(device)
        decoder_input = self.embeddings(decoder_input).view(-1,1,self.embedding_dim)
     

        max_target_len = answers.size(1)
        current_input = decoder_input
        current_hidden = context_vector.unsqueeze(1).transpose(0,1)

        total_loss = torch.tensor(0.0)
        teacher_forcing_ratio = 0.3
        import random
        preidct_each_t = []
        for t in range(max_target_len):

            ot,ht,probs,predicts = self.predict_step(current_input,current_hidden)

            # label :N
            label = answers[:,t]
            # mask : N
            mask = ans_masks[:,t]
            loss,nTotal = self.maskNLLLoss(probs,label,mask)
            total_loss+=loss
            preidct_each_t.append(predicts)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                current_input = self.embeddings(label).unsqueeze(1)
            else:
                current_input = self.embeddings(predicts)    
            current_hidden = ht

        # from tensorflow official cite
        #It's worth pointing out that we divide the loss by batch_size, 
        #so our hyperparameters are "invariant" to batch_size. 
        #Some people divide the loss by (batch_size * num_time_steps), 
        #which plays down the errors made on short sentences. 
        avg_loss = total_loss/batch_size
        preidct_each_t = torch.cat(preidct_each_t,1)


        return avg_loss,preidct_each_t





