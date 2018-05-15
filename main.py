from utils import generate_examples,expression2Xs,answer2Ys
import pickle as pkl
from train import Trainer
import torch,data,argparse
from model import Seq2seqCalc
import evaluate




def parse():
     parser = argparse.ArgumentParser(description='Attention Seq2Seq Calculator')



     parser.add_argument('mode',default="no")
     parser.add_argument('-bs', '--batch_size', help='generate data',default=32,type=int)
     parser.add_argument('-ep', '--epoch_num', help='generate data',default=50,type=int)
     parser.add_argument('-cpp', '--checkpoint_path', help='generate data')

     parser.add_argument('-g', '--generate_data', help='generate data',action='store_true')
     parser.add_argument('-gp', '--generate_path', help='generate data',default='./datas.pkl')
     parser.add_argument('-gn', '--generate_num', help='generate data',type=int)

     parser.add_argument('-ld', '--load_data', help='load data',action='store_true')
     parser.add_argument('-dp', '--data_path')

     parser.add_argument('-lm', '--load_model', help='load  model',action='store_true')
     parser.add_argument('-mp', '--model_path')
     parser.add_argument('-lr', '--lr')
     args = parser.parse_args()
     return args

from torch import optim
def run(args):
    vocab = data.Vocab() 
    
    if args.generate_data:
        generate_train_val_test(args.generate_num,vocab,0.7,0.2,args.generate_path)
        return



    batch_size = args.batch_size
    if args.load_data == True:
        data_path = args.data_path
        with open(data_path,'rb') as f:
            train_questions,train_ans,val_questions,val_ans,test_questions,test_ans = pkl.load(f)
            train_generator = data.BatchGenerator(train_questions, train_ans, batch_size)
            val_generator = data.BatchGenerator(val_questions, val_ans, batch_size)



    lr = 0.01 
    if args.load_model == True:
        model_path = args.model_path
        checkpoint = torch.load(model_path)
        seq2seq = Seq2seqCalc(*checkpoint['model_hyper'])
        seq2seq.load_state_dict(checkpoint['model']) 
        optimizer = optim.Adam(seq2seq.parameters())
        optim.Adam(seq2seq.parameters()).load_state_dict(checkpoint['optimzer'])
    else:
        #create new model
        embedding_dim,vocab_size,digit_rnn_units,decoder_rnn_units = 32,vocab .size(),256,128
        optimizer = optim.Adam(seq2seq.parameters(),lr=lr)
        seq2seq = Seq2seqCalc(embedding_dim,vocab_size,digit_rnn_units,decoder_rnn_units)
        
       
    if args.mode == 'train':
        assert optimizer is not None
        trainer = Trainer(seq2seq, optimizer ,args.epoch_num)
        trainer.train( train_generator,val_generator,vocab,lr,10,10,args.checkpoint_path)
    elif args.mode=='test':
        eva_generator = data.BatchGenerator(train_questions, train_ans, len(train_questions))
        evaluate.evaluate(seq2seq, eva_generator, vocab)


def generate_train_val_test(example_num,vocab,train_rate,val_rate,save_path):
    train_num = int(example_num*train_rate)
    val_num = int(example_num*val_rate)
    all_question,all_ans = generate_examples(example_num)
    print(all_question)
    train_questions,train_ans = all_question[:train_num],all_ans[:train_num]
    val_questions,val_ans = all_question[train_num:train_num+val_num],all_ans[train_num:train_num+val_num]
    test_questions,test_ans = all_question[train_num+val_num:],all_ans[train_num+val_num:]
    train_questions,train_ans = expression2Xs(train_questions,vocab),answer2Ys(train_ans,vocab)
    val_questions,val_ans = expression2Xs(val_questions,vocab),answer2Ys(val_ans,vocab) 
    test_questions,test_ans = expression2Xs(test_questions,vocab),answer2Ys(test_ans,vocab) 
    print(train_questions)
    print(train_ans)

    with open(save_path,'wb') as f :
        pkl.dump((train_questions,train_ans,val_questions,val_ans,test_questions,test_ans),f)

    return train_questions,train_ans,val_questions,val_ans,test_questions,test_ans


#train_questions,train_ans,val_questions,val_ans,test_questions,test_ans =\



if __name__ == '__main__':
    #vocab = data.Vocab() 
    #generate_train_val_test(10,vocab,0.6,0.2,'./datas.pkl')
    args = parse()
    run(args)
#print((train_questions,train_ans,val_questions,val_ans,test_questions,test_ans))