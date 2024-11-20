from sympy import true
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from utils.dice_score import dice_loss
from    copy import deepcopy
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.scheduler.lr_scheduler import WarmupPolyLR

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, model):
        """
        :param args:
        """
        super(Meta, self).__init__()
        self.args = args
        self.current_epoch = 0
        self.use_multi_step_loss_optimization = args.use_importance
        self.outer_loop_lr = args.update_lr
        self.meta_lr = args.meta_lr
        # self.n_way = args.n_way
        # self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.update_step = args.update_step
        self.model = model
        self.outer_loop_optimizer = optim.RMSprop(self.model.parameters(),lr=self.outer_loop_lr, weight_decay=1e-8, momentum=0.999, foreach=True)

        # print("Inner Loop parameters")
        # for key, value in self.inner_loop_optimizer.named_parameters():
        #     print(key, value.shape)

    def forward(self, args, x_spt, y_spt, x_qry, y_qry, training_phase=True, epoch=0):
        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_qry))]
        self.current_epoch = epoch
        self.model.zero_grad()
        #torch.Size([4, 3, 5, 3, 400, 400]) torch.Size([4, 3, 5, 400, 400]) torch.Size([4, 3, 1, 3, 400, 400]) torch.Size([4, 3, 1, 400, 400])
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(zip(x_spt,y_spt,x_qry,y_qry)):
            #torch.Size([3, 5, 3, 400, 400]) torch.Size([3, 5, 400, 400])
            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1, h, w)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1, h, w)

            for num_step in range(self.update_step):
                spt_logit = self.model(x_support_set_task) #torch.Size([3, 1, 400, 400])
                support_loss = dice_loss(torch.sigmoid(spt_logit.squeeze(1)), y_support_set_task.float(), multiclass=False)
                
                if args.second_order == "True":
                    args.second_order = 1
                    
                grad = torch.autograd.grad(support_loss, self.model.parameters(), create_graph=args.second_order, allow_unused=True)
                updated_params = {}
                for (name,param), grad in zip(self.model.named_parameters(), grad):
                    if grad is None:
                        grad = 0
                    updated_params[name] = (param - self.meta_lr * grad)
                
                self.model.module.update_params(updated_params)

                if self.use_multi_step_loss_optimization and training_phase and epoch < args.multi_step_loss_num_epochs:
                    target_preds = self.model(x_target_set_task)
                    target_loss = dice_loss(torch.sigmoid(target_preds.squeeze(1)), y_target_set_task.float(), multiclass=False)
                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                elif num_step == (self.update_step - 1):
                    target_preds = self.model(x_target_set_task)
                    target_loss = dice_loss(torch.sigmoid(target_preds.squeeze(1)), y_target_set_task.float(), multiclass=False)
                    task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)
            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                self.model.restore_backup_stats()
             
        losses = self.get_across_task_loss_metrics(total_losses=total_losses, total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()
        
        loss=losses['loss']
        self.outer_loop_optimizer.zero_grad()
        loss.backward()
        self.outer_loop_optimizer.step()
        self.outer_loop_optimizer.zero_grad()
        self.zero_grad()
        
        return losses, per_task_target_preds
    
    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = {'loss': torch.mean(torch.stack(total_losses))}
        
        losses['accuracy'] = torch.mean(torch.stack(total_accuracies)) #np.mean(total_accuracies)

        return losses
    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.update_step)) * (
                1.0 / self.args.update_step)
        decay_rate = 1.0 / self.args.update_step / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.update_step
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.update_step - 1) * decay_rate),
            1.0 - ((self.args.update_step - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).cuda()
            
        return loss_weights