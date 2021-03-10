if __name__ == '__main__':
    # if config.mode not in ['f_d', 'implication', 'pr_loss', 'gcross', 'learn2reweight','label_snorkel', 'pure_snorkel', 'gcross_snorkel']:
    if config.mode not in ['implication','learn2reweight','cage']:
        raise ValueError('Invalid run mode ' + config.mode)