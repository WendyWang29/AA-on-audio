'''
*** Adversarial attacks test script ***
These are not the attacks to be used on large scale (batches of data)
Instead these should be used on single files, to test if they work

Attacks implemented:
- FGSM untargeted

author: wwang
'''

class untargeted_FGSM_single_file(object):
    def __init__(self, model, epsilon, data, label, device):
        self.model = model
        self.epsilon = epsilon
        self.data = data
        self.label = label
        self.device = device

    def run_attack(self):
        # run the attack on one single file

        data = self.data.to(self.device)
        label = self.label.to(self.device)

        # FGSM requires grad wrt the data
        data.requires_grad = True

        out = self.model(data)



