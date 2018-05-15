import torch,os
from tqdm import tqdm

import evaluate
class Trainer():
    def __init__(self,model,optimizer,epoch_num):
        self.epoch_num = epoch_num
        self.model = model
        self.optimizer = optimizer
        


    
    def train(self,train_generator,val_generator,vocab,lr,eval_every,save_every,save_dir):
       
        

        for epoch in tqdm(range(self.epoch_num)):
            print('Epoch :%d '%(epoch))
            for Xs,ys in train_generator.get_batches():
                self.model.zero_grad()
                loss,predict_max = self.model(Xs,ys)
                loss.backward()

                clip = 50.0
                torch.nn.utils.clip_grad_norm(self.model.parameters(), clip)
                self.optimizer.step()
            
            print('loss is :%.3f'%(loss.item())) 

            if epoch % save_every == 0: 
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save({'epoch':str(epoch),'model':self.model.state_dict(),'optimzer':self.optimizer.state_dict(),'model_hyper':self.model.get_hyper()},os.path.join(save_dir,'model_%d.pkl'%(epoch)))

            if epoch%eval_every == 0:
                print('train accuracy')
                evaluate.evaluate(self.model,train_generator,vocab)
                print('validation accuracy')
                evaluate.evaluate(self.model,val_generator,vocab)
            lr = lr*0.99
        print('evaluate')
        evaluate.evaluate(self.model,train_generator,vocab)
        torch.save({'epoch':str(epoch),'model':self.model.state_dict(),'optimzer':self.optimizer.state_dict(),'model_hyper':self.model.get_hyper()},os.path.join(save_dir,'model_%d.pkl'%(epoch)))