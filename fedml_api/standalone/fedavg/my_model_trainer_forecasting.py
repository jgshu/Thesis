import torch
import torch.nn as nn
import logging

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    # from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    pass

class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.MSELoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
            #                              weight_decay=args.wd, amsgrad=True)
            # 未正则化
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
            # 正则化
            # optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        epoch_loss = []

        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Epoch: {}\tLoss: {:.6f}'.format(
                epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_loss': 0,
            'test_total': 0
        }
        criterion = nn.MSELoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
