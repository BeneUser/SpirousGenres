import copy
import random
import torch

#Test augments
def no_augment(audio_samples):
    return audio_samples

def zero(audio_samples):
    return torch.zeros(audio_samples.size(0))


#UPDATE WHEN IMPLEMENTING MORE AUGMENTATIONS!
#ORDER IS IMPORTANT
all_augmentations = [no_augment, zero]


#Var do_augments: apply
def augment(x, augments):
    #Apply the chosen augmentations
    for a in augments:
        x = a(x)
    return x
        
def choose_augmentations(num_augments, fixed_augments):
    #Make sure that inputs are valid
    if fixed_augments != None:
        assert len(fixed_augments) != all_augmentations, "always_augment list doesn't have the same size as list of all augmentations"
    else:
        fixed_augments = [0] * len(all_augmentations)
    assert num_augments <= len(all_augmentations), "There aren't that many augmentations available!"

    #Randomize which augmentations to use:
    #Initialize bit map of which augmentations to use.
    do_augments = copy.deepcopy(fixed_augments)
    #Free range is the number of augmentations, that we can still add.
    freerange = len(all_augmentations) - sum(fixed_augments)
    for i in range(num_augments - sum(fixed_augments)):
        #Choose a pseudo position for additional augmentation.
        chosen = random.randint(0, freerange-1)
        #Find out the actual position p
        pseudo_p = 0
        p = 0
        while(pseudo_p != chosen or all_augmentations[p] == 1):
            if all_augmentations[p] == 0:
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
            augments.append(all_augmentations[i])
    return augments