from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch,data

seq_lens = [3,2,4,5,1]
seqs = [[[1,1],[2,2],[3,3]],
       [[4,4],[5,5]],
       [[6,6],[7,7],[8,8],[9,9]],
       [[-1,-1],[-2,-2],[0,0],[-2,-2],[0,0]],
       [[33,33]]]

seq_lens = torch.tensor(seq_lens,dtype=torch.long)
seq_tensors =  torch.zeros((seq_lens.size(0),seq_lens.max(),2),dtype=torch.long)
#seqs = torch.tensor(seqs,dtype=torch.long)
for i,(seq,seq_len) in enumerate(zip(seqs, seq_lens)):
    seq_tensors[i,:seq_len,:] = torch.tensor(seq,dtype=torch.long)

_seq_tensors = seq_tensors

seq_lens, perm_idx = seq_lens.sort(0, descending=True)
#print(perm_idx)
seq_tensors  =  seq_tensors [perm_idx]
#seq length: type is not a tensor
packed_input = pack_padded_sequence(seq_tensors, seq_lens.data.cpu().numpy(),batch_first=True)
#print(packed_input)
#packed_output, ht = rnn_cell(packed_input,hts)
pad_output,plens = pad_packed_sequence(packed_input,batch_first=True)
#print(pad_output)
#print(plens)
_,unsort_idx= perm_idx.sort(0)
#print(unsort_idx)
output = pad_output[unsort_idx]
#print(output)

assert torch.equal(output ,_seq_tensors)