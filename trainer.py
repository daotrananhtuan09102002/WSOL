from tqdm import tqdm
import wsol
import wsol.method
import os
import torch
import torch.nn as nn


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['fc.']
    }

    def __init__(self, dataset_name, architecture, architecture_type, pretrained,
                 large_feature_map, drop_threshold, drop_prob, lr, lr_classifier_ratio,
                 momentum, weight_decay, lr_decay_points, lr_decay_rate,
                 sim_fg_thres, sim_bg_thres, loss_ratio_drop,
                 loss_ratio_sim, loss_ratio_norm, wsol_method, loader, log_dir):
        self.dataset_name = dataset_name
        self.architecture = architecture
        self.architecture_type = architecture_type
        self.pretrained = pretrained
        self.large_feature_map = large_feature_map
        self.drop_threshold = drop_threshold
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_classifier_ratio = lr_classifier_ratio
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay_points = lr_decay_points
        self.lr_decay_rate = lr_decay_rate

        self.sim_fg_thres = sim_fg_thres
        self.sim_bg_thres = sim_bg_thres
        self.loss_ratio_drop = loss_ratio_drop
        self.loss_ratio_sim = loss_ratio_sim
        self.loss_ratio_norm = loss_ratio_norm
        self.wsol_method = wsol_method

        self.model = self._set_model()
        self.model_multi = torch.nn.DataParallel(self.model)
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.l1_loss = nn.L1Loss().cuda()
        self.optimizer = self._set_optimizer()

        self.loader = loader
        self.log_dir = log_dir


    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.dataset_name]
        print("Loading model {}".format(self.architecture))
        arch = self.architecture
        model = wsol.__dict__[arch](
            dataset_name=self.dataset_name,
            architecture_type=self.architecture_type,
            pretrained=self.pretrained,
            num_classes=num_classes,
            large_feature_map=self.large_feature_map,
            drop_threshold=self.drop_threshold,
            drop_prob=self.drop_prob)
        model = model.cuda()
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []
        param_features_name = []
        param_classifiers_name = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        def string_contains_any(string, substring_list):
            for substring in substring_list:
                if substring in string:
                    return True
            return False

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.architecture)):
                if self.architecture == 'vgg16':
                    param_features.append(parameter)
                    param_features_name.append(name)
                elif self.architecture == 'resnet50':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
            else:
                if self.architecture == 'vgg16':
                    param_classifiers.append(parameter)
                    param_classifiers_name.append(name)
                elif self.architecture == 'resnet50':
                    param_features.append(parameter)
                    param_features_name.append(name)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.lr},
            {'params': param_classifiers,
             'lr': self.lr * self.lr_classifier_ratio}],
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True)
        return optimizer

    def _get_loss_alignment(self, feature, sim, target, eps=1e-15):

        def normalize_minmax(cams, eps=1e-15):
            """
            Args:
                cam: torch.Tensor(size=(B, H, W), dtype=np.float)
            Returns:
                torch.Tensor(size=(B, H, W), dtype=np.float) between 0 and 1.
                If input array is constant, a zero-array is returned.
            """

            B, _, _ = cams.shape
            min_value, _ = cams.view(B, -1).min(1)
            cams_minmax = cams - min_value.view(B, 1, 1)
            max_value, _ = cams_minmax.view(B, -1).max(1)
            cams_minmax /= max_value.view(B, 1, 1) + eps
            return cams_minmax

        B = target.size(0)
        feature_norm = torch.norm(feature, dim=1)
        feature_norm_minmax = normalize_minmax(feature_norm)
        sim_target_flat = sim[torch.arange(B), target].view(B, -1)
        feature_norm_minmax_flat = feature_norm_minmax.view(B, -1)
        if self.dataset_name == 'ILSVRC':
            sim_fg = (feature_norm_minmax_flat > self.sim_fg_thres).float()
            sim_bg = (feature_norm_minmax_flat < self.sim_bg_thres).float()

            sim_fg_mean = (sim_fg * sim_target_flat).sum(dim=1) / (sim_fg.sum(dim=1) + eps)
            sim_bg_mean = (sim_bg * sim_target_flat).sum(dim=1) / (sim_bg.sum(dim=1) + eps)
            loss_sim = torch.mean(sim_bg_mean - sim_fg_mean)

            norm_fg = (sim_target_flat > 0).float()
            norm_bg = (sim_target_flat < 0).float()

            norm_fg_mean = (norm_fg * feature_norm_minmax_flat).sum(dim=1) / (norm_fg.sum(dim=1) + eps)
            norm_bg_mean = (norm_bg * feature_norm_minmax_flat).sum(dim=1) / (norm_bg.sum(dim=1) + eps)

            loss_norm = torch.mean(norm_bg_mean - norm_fg_mean)
        elif self.dataset_name == 'CUB':
            sim_fg = (feature_norm_minmax_flat > self.sim_fg_thres).float()
            sim_bg = (feature_norm_minmax_flat < self.sim_bg_thres).float()

            sim_fg_mean = (sim_fg * sim_target_flat).sum(dim=1) / (sim_fg.sum(dim=1) + eps)
            sim_bg_mean = (sim_bg * sim_target_flat).sum(dim=1) / (sim_bg.sum(dim=1) + eps)
            loss_sim = torch.mean(sim_bg_mean - sim_fg_mean)

            sim_max_class, _ = sim.max(dim=1)
            sim_max_class_flat = sim_max_class.view(B, -1)

            norm_fg = (sim_max_class_flat > 0).float()
            norm_bg = (sim_max_class_flat < 0).float()

            norm_fg_mean = (norm_fg * feature_norm_minmax_flat).sum(dim=1) / (norm_fg.sum(dim=1) + eps)
            norm_bg_mean = (norm_bg * feature_norm_minmax_flat).sum(dim=1) / (norm_bg.sum(dim=1) + eps)

            loss_norm = torch.mean(norm_bg_mean - norm_fg_mean)
        else:
            raise ValueError("dataset_name should be in ['ILSVRC', 'CUB']")

        return loss_sim, loss_norm

    def _wsol_training(self, images, target, warm=False):
        output_dict = self.model_multi(images, labels=target)
        logits = output_dict['logits']

        if self.wsol_method == 'bridging-gap':
            loss_ce = self.cross_entropy_loss(logits, target)

            loss_drop = self.l1_loss(output_dict['feature'], output_dict['feature_erased'])
            loss_sim, loss_norm = \
                self._get_loss_alignment(output_dict['feature'], output_dict['sim'], target)

            loss = loss_ce + self.loss_ratio_drop * loss_drop
            if not warm:
                loss += self.loss_ratio_sim * loss_sim + self.loss_ratio_norm * loss_norm
        elif self.wsol_method == 'cam':
            loss = self.cross_entropy_loss(logits, target)
        else:
            raise ValueError("wsol_method should be in ['bridging-gap', 'cam']")

        return logits, loss


    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch in self.lr_decay_points:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_decay_rate


    def _torch_save_model(self, filename):
        torch.save({'state_dict': self.model.state_dict()},
                   os.path.join(self.log_dir, filename))


    def save_checkpoint(self, epoch, checkpoint_path):
        print("Saving checkpoint to {}".format(checkpoint_path))
        self._torch_save_model(
            f'{checkpoint_path}{epoch}_checkpoint.pth.tar')
        
    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(
            self.log_dir,
            checkpoint_name)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))


    def train(self, warm=False):
        self.model_multi.train()
        loader = self.loader

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            logits, loss = self._wsol_training(images, target, warm=warm)
            pred = logits.argmax(dim=1)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        return dict(classification_acc=classification_acc, loss=loss_average)