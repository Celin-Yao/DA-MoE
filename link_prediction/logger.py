import torch
import os
import numpy as np

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self,dataset,epochs,path,model_type,gate_model,num_layers,min_layers,hidden_dim,dropout,coef,topK,seed,run=None):
        log_folder_name = os.path.join(*[dataset])

        if not(os.path.isdir('./{}/{}'.format(path,log_folder_name))):
            os.makedirs(os.path.join('./{}/{}'.format(path,log_folder_name)))
        
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()

            final_total_result_file = "./{}/{}/{}-total_results.txt".format(path,dataset, "sag")

            summary_1_total = ' Seed={},Dataset={}, model_type={},gate_model={},epochs={},num_layers={},min_layers={},hidden_dim={},dropout={},topK={},coef={}'.format(
                     run+seed,dataset,model_type,gate_model,epochs,num_layers,min_layers,hidden_dim,dropout,topK,coef)
            summary_2_total = 'train_acc_main={:.2f},test_acc_total_lst={:.2f}'.format(
                   result[argmax, 0],result[argmax, 2])
            with open(final_total_result_file, 'a+') as f:
                f.write('{} : \n{} \n{}: \n{}\n{}\n'.format(
                        'Model details:',
                        summary_1_total,
                        "Final result:",
                        summary_2_total,
                        "-"*30)
                        )
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            final_total_result_file = "./{}/{}/{}-total_seeds_results.txt".format(path,dataset, "sag")

            summary_1_total = ' Dataset={},model_type={},,gate_model={}, epochs={},num_layers={},min_layers={},hidden_dim={},dropout={},topK={},coef={}'.format(
                        dataset,model_type,gate_model,epochs,num_layers,min_layers,hidden_dim,dropout,topK,coef)
            summary_2_total = 'train_acc_main={:.2f} ± {:.4f},test_acc_total_lst={:.2f} ± {:.4f}'.format(
                    best_result[:, 2].mean(),best_result[:, 2].std(),best_result[:, 3].mean(),best_result[:, 3].std())
            
            with open(final_total_result_file, 'a+') as f:
                f.write('{} : \n{} \n{}: \n{}\n{}\n'.format(
                        'Model details:',
                        summary_1_total,
                        "Final result:",
                        summary_2_total,
                        "-"*30)
                        )
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')