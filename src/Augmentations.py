import copy
import random
import torch


class Augmentations():
    def __init__(self, available_augmentations, num_augments, always_augment):
        #In principal, all of these augmentations may be used
        self.available_augmentations = list(self.__getattribute__(a) for a in available_augmentations)

        #Local variables for choosing how to augment
        if always_augment != None:
            assert len(always_augment) == len(self.available_augmentations), "always_augment list doesn't have the same size as list of available augmentations"
        else:
            always_augment = [0] * len(self.available_augmentations)
        assert num_augments <= len(self.available_augmentations), "num_augments larger than the number of available augmentations!"
        self.num_augments = num_augments #Number of total augmentations (including those fixed in always_augment)
        self.fixed_augments = always_augment #Which augmentations to always use

        #Other local vars, eg such need for certain augmentations to work.
        #...

    #AUGMENTATIONS
    #Test augmentations
    def no_augment(self, audio_samples):
        return audio_samples

    def zero(self, audio_samples):
        return torch.zeros(audio_samples.size(0))


    def augment(self, x):
        augments = self.choose_augmentations()
        #print("Chosen augmentations [" + " -> ".join(a.__name__ for a in self.augmentations) +"]")
        #Apply the chosen augmentations
        for a in augments:
            x = a(x)
        return x
            
    def choose_augmentations(self):
        #Randomize which augmentations to use:
        #Initialize bit map of which augmentations to use.
        do_augments = copy.deepcopy(self.fixed_augments)
        #Free range is the number of augmentations, that we can still add.
        freerange = len(self.available_augmentations) - sum(self.fixed_augments)
        for i in range(self.num_augments - sum(self.fixed_augments)):
            #Choose a pseudo position for additional augmentation.
            chosen = random.randint(0, freerange-1)
            #Find out the actual position p
            pseudo_p = 0
            p = 0
            while(pseudo_p != chosen or self.available_augmentations[p] == 1):
                if self.available_augmentations[p] == 0:
                    pseudo_p += 1 #Only skip if the augmentation is not used yet.
                p += 1
            #Set the augmentation to true
            do_augments[p] = 1
            #Update the free range of pseudo positions
            freerange -= 1

        #Collect and return the chosen augmentations
        augments = []
        for i in range(len(do_augments)):
            if do_augments[i] == 1:
                augments.append(self.available_augmentations[i])
        return augments